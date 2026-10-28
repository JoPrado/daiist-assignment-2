"""
Tiling shared by train.ipynb and app.ipynb.

Every image in the Roboflow dataset contains at least one defect, so there is
no defect-free fabric to measure false alarms on. Cutting each 640x640 image
into a grid of tiles gives tiles with no defect in them: clean fabric from the
same cameras and looms. A tile is the unit the model inspects and the unit the
pass / review / fail decision is made on.

Keeping this in one place guarantees the dashboard cuts an uploaded image
exactly the way the training and test tiles were cut.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_SIZE = 640  # every dataset image is 640x640; uploads are resized to it
GRID = 2  # 2x2 grid -> 320x320 tiles
TILE_SIZE = IMAGE_SIZE // GRID
# A box cut by a tile edge keeps its label in that tile only if at least this
# share of its area falls inside. A tile that holds only smaller slivers is
# neither clearly clean nor clearly defective, so it is left out entirely.
MIN_VISIBLE = 0.25
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # folder -> data.yaml key


def source_stem(filename: str) -> str:
    """'014885_jpg.rf.17d7...jpg' -> '014885': the original photo before Roboflow's copies."""
    return filename.split(".rf.")[0].rsplit("_", 1)[0]


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """YOLO label file -> (class ids, boxes as pixel xyxy). Polygons become their bounding box."""
    classes, boxes = [], []
    for line in path.read_text().splitlines():
        values = line.split()
        if not values:
            continue
        classes.append(int(values[0]))
        coords = np.array(values[1:], dtype=float)
        if len(coords) == 4:
            xc, yc, w, h = coords
            box = [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
        else:
            xy = coords.reshape(-1, 2)
            box = [*xy.min(axis=0), *xy.max(axis=0)]
        boxes.append(np.clip(box, 0, 1) * IMAGE_SIZE)
    return np.array(classes, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def cut_tiles(image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Resize to IMAGE_SIZE and cut into GRID x GRID tiles: [(row, col, tile), ...]."""
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return [
        (r, c, image[r * TILE_SIZE:(r + 1) * TILE_SIZE, c * TILE_SIZE:(c + 1) * TILE_SIZE])
        for r in range(GRID)
        for c in range(GRID)
    ]


def boxes_in_tile(classes: np.ndarray, boxes: np.ndarray, row: int, col: int):
    """
    Clip the image's boxes to one tile.

    Returns (classes, boxes in tile pixel xyxy, ambiguous). `ambiguous` is True
    when some box overlaps the tile but less than MIN_VISIBLE of it is inside.
    """
    x0, y0 = col * TILE_SIZE, row * TILE_SIZE
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(boxes[:, [0, 2]], x0, x0 + TILE_SIZE) - x0
    clipped[:, [1, 3]] = np.clip(boxes[:, [1, 3]], y0, y0 + TILE_SIZE) - y0

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    inside = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    visible = np.divide(inside, area, out=np.zeros_like(area), where=area > 0)

    keep = visible >= MIN_VISIBLE
    ambiguous = bool(((inside > 0) & ~keep).any())
    return classes[keep], clipped[keep], ambiguous


def to_yolo_lines(classes: np.ndarray, boxes: np.ndarray) -> str:
    lines = []
    for c, (x1, y1, x2, y2) in zip(classes, boxes / TILE_SIZE):
        lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return "\n".join(lines)


def build_tiled_dataset(src: Path, dst: Path, names: list[str]) -> pd.DataFrame:
    """
    Tile every split of a YOLO-format dataset from `src` into `dst`.

    Train images whose original photo also appears in valid or test are
    dropped, so no evaluation tile has a near-copy in training. Returns one
    row per tile written: split, file, and the number of boxes of each class.
    """
    held_out = {
        source_stem(p.name)
        for folder in ["valid", "test"]
        for p in (src / folder / "images").iterdir()
    }
    rows = []
    for folder in SPLITS:
        (dst / folder / "images").mkdir(parents=True, exist_ok=True)
        (dst / folder / "labels").mkdir(parents=True, exist_ok=True)
        for image_path in sorted((src / folder / "images").iterdir()):
            stem = source_stem(image_path.name)
            if folder == "train" and stem in held_out:
                rows.append({"split": folder, "file": None, "source": stem, "status": "leak"})
                continue
            classes, boxes = read_labels(src / folder / "labels" / f"{image_path.stem}.txt")
            image = cv2.imread(str(image_path))
            for r, c, tile in cut_tiles(image):
                tile_classes, tile_boxes, ambiguous = boxes_in_tile(classes, boxes, r, c)
                name = f"{image_path.stem}_r{r}c{c}"
                row = {"split": folder, "file": name, "source": stem}
                if ambiguous:
                    rows.append(row | {"status": "ambiguous"})
                    continue
                cv2.imwrite(str(dst / folder / "images" / f"{name}.jpg"), tile)
                (dst / folder / "labels" / f"{name}.txt").write_text(to_yolo_lines(tile_classes, tile_boxes))
                counts = {n: int((tile_classes == i).sum()) for i, n in enumerate(names)}
                rows.append(row | {"status": "kept", **counts})

    yaml = [f"path: {dst.resolve().as_posix()}"]
    yaml += [f"{key}: {folder}/images" for folder, key in SPLITS.items()]
    yaml += [f"names: {names}"]
    (dst / "data.yaml").write_text("\n".join(yaml) + "\n")

    manifest = pd.DataFrame(rows)
    for n in names:
        manifest[n] = manifest.get(n, 0).fillna(0).astype(int)
    manifest["defect"] = manifest[names].sum(axis=1) > 0
    return manifest
