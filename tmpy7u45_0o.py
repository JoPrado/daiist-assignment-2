import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

from costing import (alternatives, decide, expected_cost, outcome_rates, per_thousand,
                     review_share, reviewers_needed)
from tiling import IMAGE_SIZE, TILE_SIZE, cut_tiles

ARTIFACTS = Path("artifacts")
FIG_DIR = ARTIFACTS / "figures"
WEIGHTS = Path("weights") / "fabric_defect_yolo11s.pt"
SCORE_FLOOR = 0.01  # same floor train.ipynb used when it scored the tiles

metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
scores = pd.read_csv(ARTIFACTS / "tile_scores.csv")
sweep = pd.read_csv(ARTIFACTS / "cost_sweep.csv")
model = YOLO(str(WEIGHTS))

ASSUMPTIONS = metrics["assumptions"]
NAMES = metrics["dataset"]["classes"]
OP = metrics["operating_point"]
test = scores[scores["split"] == "test"]
TEST_SCORES = test["score_finetuned"].to_numpy()
TEST_DEFECT = test["defect"].to_numpy().astype(bool)

COLOURS = {"pass": (46, 160, 67), "review": (230, 160, 0), "fail": (210, 40, 40)}  # RGB


def inspect(image: np.ndarray, t_low: float, t_high: float):
    if image is None:
        return None, pd.DataFrame()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # tiles were written by OpenCV, so the model saw BGR
    tiles = cut_tiles(bgr)
    results = model.predict([t for _, _, t in tiles], imgsz=TILE_SIZE, conf=SCORE_FLOOR, verbose=False)
    canvas = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    rows = []
    for (r, c, _), res in zip(tiles, results):
        conf, cls, xyxy = res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.xyxy.cpu().numpy()
        score = float(conf.max(initial=0.0))
        outcome = str(decide(np.array([score]), t_low, t_high)[0])
        x0, y0 = c * TILE_SIZE, r * TILE_SIZE
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + TILE_SIZE - 3, y0 + TILE_SIZE - 3), COLOURS[outcome], 4)
        cv2.putText(canvas, f"{outcome} {score:.2f}", (x0 + 8, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[outcome], 2)
        shown = conf >= min(t_low, t_high)  # draw only the boxes that drive a review or fail
        for k, (x1, y1, x2, y2) in zip(np.flatnonzero(shown), xyxy[shown]):
            p1, p2 = (int(x0 + x1), int(y0 + y1)), (int(x0 + x2), int(y0 + y2))
            cv2.rectangle(canvas, p1, p2, (255, 255, 255), 2)
            cv2.putText(canvas, f"{NAMES[cls[k]]} {conf[k]:.2f}", (p1[0], max(p1[1] - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        best = NAMES[cls[conf.argmax()]] if len(conf) else "-"
        rows.append({"tile": f"row {r}, col {c}", "score": round(score, 3), "top class": best, "decision": outcome})
    return canvas, pd.DataFrame(rows)

def operating_view(t_low: float, t_high: float, base_rate: float):
    t_low = min(t_low, t_high)
    assumptions = ASSUMPTIONS | {"base_rate": base_rate}
    rates = outcome_rates(TEST_SCORES, TEST_DEFECT, t_low, t_high)
    counts = per_thousand(rates, base_rate)
    matrix = pd.DataFrame(
        [[truth] + [round(counts[f"{o}|{key}"], 1) for o in ["pass", "review", "fail"]]
         for truth, key in [("defective tile", "defect"), ("clean tile", "clean")]],
        columns=["truth \\ decision", "pass", "review", "fail"])
    raw = pd.DataFrame(
        [[truth] + [int(((decide(TEST_SCORES, t_low, t_high) == o) & mask).sum()) for o in ["pass", "review", "fail"]]
         for truth, mask in [("defective tile", TEST_DEFECT), ("clean tile", ~TEST_DEFECT)]],
        columns=["truth \\ decision", "pass", "review", "fail"])

    cost = expected_cost(rates, assumptions)
    share = review_share(rates, base_rate)
    people = reviewers_needed(share, assumptions)
    summary = (f"### EUR {1000 * cost:,.2f} per 1,000 tiles\n"
               f"- Sent to review: **{100 * share:.1f}%** of tiles, "
               f"{share * assumptions['tiles_per_day']:,.0f} a day, **{people:.2f} reviewers** "
               f"at {assumptions['review_seconds']} s a tile and {assumptions['shift_hours']} h shifts\n"
               f"- Defects caught (fail or review): **{100 * (rates['fail|defect'] + rates['review|defect']):.1f}%**, "
               f"shipped: {100 * rates['pass|defect']:.1f}%\n"
               f"- Clean fabric pulled without review: {100 * rates['fail|clean']:.1f}%")

    designs = {name: c for name, c in alternatives(assumptions).items()}
    single = outcome_rates(TEST_SCORES, TEST_DEFECT, OP["threshold"], OP["threshold"])
    designs[f"model alone (threshold {OP['threshold']:.2f})"] = expected_cost(single, assumptions)
    designs[f"this band ({t_low:.2f}-{t_high:.2f})"] = cost
    compare = pd.DataFrame({"design": list(designs), "EUR per 1,000 tiles": [round(1000 * v, 2) for v in designs.values()],
                            "EUR per day": [round(v * assumptions["tiles_per_day"], 0) for v in designs.values()]})
    return matrix, raw, summary, compare, cost_curve(t_low, t_high)


def cost_curve(t_low: float, t_high: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=sweep["threshold"], y=1000 * sweep["valid_cost_per_tile"], name="finetuned, validation")
    fig.add_scatter(x=sweep["threshold"], y=1000 * sweep["test_cost_per_tile"], name="finetuned, test")
    fig.add_scatter(x=sweep["threshold"], y=1000 * sweep["test_coco_cost_per_tile"], name="COCO zero-shot, test",
                    line={"dash": "dot"})
    for name, c in alternatives(ASSUMPTIONS).items():
        fig.add_hline(y=1000 * c, line={"dash": "dash", "color": "grey", "width": 1},
                      annotation_text=name, annotation_position="top left")
    fig.add_vline(x=OP["threshold"], line={"color": "black"},
                  annotation_text=f"chosen {OP['threshold']:.2f}", annotation_position="top right")
    fig.add_vrect(x0=t_low, x1=max(t_high, t_low + 1e-3), fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="review band", annotation_position="bottom left")
    fig.update_layout(xaxis_title="confidence threshold (single cut-off, no review)",
                      yaxis_title="expected EUR per 1,000 tiles", height=420, margin={"t": 30},
                      legend={"orientation": "h", "y": -0.25})
    return fig

per_class = pd.DataFrame(metrics["per_class"]).round(3)
baselines = pd.DataFrame(metrics["baselines"]).round(3)
cm = metrics["confusion_matrix"]
box_confusion = pd.DataFrame(cm["matrix"], index=[f"pred {l}" for l in cm["labels"]],
                             columns=[f"true {l}" for l in cm["labels"]]).reset_index(names="")
assumption_rows = pd.DataFrame(
    [{"item": f"cost of {k.upper()}", "value": f"EUR {v}"} for k, v in ASSUMPTIONS["cost"].items()]
    + [{"item": k.replace("_", " "), "value": v} for k, v in ASSUMPTIONS.items() if k != "cost"])

with gr.Blocks(title="Fabric defect inspection") as demo:
    gr.Markdown("# Fabric defect inspection\n"
                "Tiles scoring below the lower threshold pass, at or above the upper one fail, "
                "and anything between goes to a person.")
    with gr.Row():
        t_low = gr.Slider(0, 1, value=OP["t_low"], step=0.01, label="lower threshold (pass below)")
        t_high = gr.Slider(0, 1, value=OP["t_high"], step=0.01, label="upper threshold (fail at or above)")
        base_rate = gr.Slider(0.005, 0.5, value=ASSUMPTIONS["base_rate"], step=0.005,
                              label="base rate: share of production tiles with a defect")

    with gr.Tab("Operating point"):
        with gr.Row():
            with gr.Column():
                summary = gr.Markdown()
                gr.Markdown("**Expected tiles per 1,000 in production** (test-split rates, reweighted to the base rate)")
                matrix = gr.Dataframe(interactive=False)
                gr.Markdown("**Raw test-split tile counts**")
                raw = gr.Dataframe(interactive=False)
            with gr.Column():
                curve = gr.Plot(label="Cost against threshold")
                compare = gr.Dataframe(label="Priced against the alternatives", interactive=False)

    with gr.Tab("Inspect an image"):
        with gr.Row():
            upload = gr.Image(type="numpy", label="Fabric image (resized to 640x640, cut into 4 tiles)")
            annotated = gr.Image(label="Decision per tile")
        tile_table = gr.Dataframe(label="Tiles", interactive=False)

    with gr.Tab("Per-class results"):
        gr.Markdown("Test split, finetuned YOLO11s. Positive class: any defect (Hole, Seam or Stitch).")
        gr.Dataframe(per_class, label="Per-class precision, recall and AP", interactive=False)
        gr.Dataframe(baselines, label="Against the baselines (same test split)", interactive=False)
        gr.Dataframe(box_confusion, label=f"Box-level confusion matrix ({cm['orientation']})", interactive=False)
        with gr.Row():
            gr.Image(str(FIG_DIR / "test_confusion_matrix_normalized.png"), label="Normalised confusion matrix")
            gr.Image(str(FIG_DIR / "test_BoxPR_curve.png"), label="Precision-recall per class")

    with gr.Tab("Assumptions"):
        gr.Dataframe(assumption_rows, interactive=False)
        gr.Image(str(FIG_DIR / "dataset.png"), label="Dataset composition")

    view = [matrix, raw, summary, compare, curve]
    for control in [t_low, t_high, base_rate]:
        control.change(operating_view, [t_low, t_high, base_rate], view)
        control.change(inspect, [upload, t_low, t_high], [annotated, tile_table])
    upload.change(inspect, [upload, t_low, t_high], [annotated, tile_table])
    demo.load(operating_view, [t_low, t_high, base_rate], view)

demo.launch()
