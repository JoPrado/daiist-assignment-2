# Assignment 2 Report
 
- **Name**: Joao Paulo T Prado
- **Student ID**: 19398
- **Email**: jtobar.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

---
 
## Business problem, and the cost of being wrong
 
**Decision supported.** For each 320×320 tile of fabric, the system chooses one of three outcomes:
 
- **pass**: the tile is shipped
- **review**: a person checks it
- **fail**: the tile is pulled

**Scoring.** Each tile's score is the highest confidence of any defect the model finds in it, or 0 if it finds none.
 
**Costs used.** These are currently set in `results.py` / `train.ipynb`:
 
| outcome | cost |
|---|---|
| missed defect shipped (FN) | €5.00 |
| clean fabric pulled (FP) | €1.00 |
| TP / TN | €0 |
| one tile reviewed by a person | €0.10 |
 
With these figures, a missed defect costs 5× a false alarm.
 
**Who acts on the decision.** Textile plants typically employ dedicated quality-control staff whose job is to stop defective products from reaching the shelves. The model is a tool to assist these workers rather than replace them: the camera system flags suspect fabric, so inspectors can focus their attention where it is needed and do their jobs more efficiently.
 
**Cost of a missed defect (FN).** When a defective product reaches the customer, the business bears an estimated €10–20 per case. That covers the logistics of the customer returning the product plus the cost of sending a replacement. The cost analysis in this report was run with FN = €5, which is below that range. The reported costs therefore understate how expensive misses are, and with a higher FN cost the cheapest threshold would be expected to move lower, flagging more tiles.
 
**Cost of a false alarm (FP).** A false alarm is a clean tile that the model wrongly flags. It is not thrown away: it goes to a quality-control employee for a manual re-check, which is estimated at about €1 per case.
 
---
 
## Dataset
 
- **Source:** Roboflow Universe, workspace `defect-detection-witqu`, project `fabric-defect-thesis`, version **1**. Link: https://universe.roboflow.com/defect-detection-witqu/fabric-defect-thesis/dataset/1
- **Classes:** Hole, Seam, Stitch.
- **Images:** each image is a 640×640 close-up of fabric.

**The problem.** Every original image contains at least one defect, so the dataset has no clean fabric to measure false alarms on.
 
**The fix.** Each image is cut into a 2×2 grid of 320 px tiles. Tiles containing no defect serve as clean fabric. The tiling follows three rules:
 
1. A box cut by a tile edge keeps its label only if at least 25% of its area falls inside the tile.
2. Tiles that hold only smaller slivers of a box are dropped as ambiguous.
3. Train images whose original photo also appears in valid or test were dropped (30 images), so that evaluation tiles have no near-copies in training.

**Why this dataset serves the problem.** The dataset contains photos of clothes and other textiles showing the kinds of defects quality control looks for, such as holes and seam faults. These are the same defect types the model would have to catch on a real line.
 
---
 
## What the dataset's composition implies
 
**Original images per split**
 
| split | images | unique originals | Hole | Seam | Stitch |
|---|---|---|---|---|---|
| train | 6568 | 2140 | 2597 | 1685 | 4499 |
| valid | 362 | 362 | 138 | 102 | 233 |
| test | 179 | 179 | 84 | 32 | 124 |
 
**After tiling**
 
| split | tiles kept | defect tiles | clean tiles | defect share | ambiguous tiles dropped |
|---|---|---|---|---|---|
| train | 17634 | 10731 | 6903 | 0.609 | 8518 |
| valid | 1036 | 589 | 447 | 0.569 | 412 |
| test | 550 | 296 | 254 | 0.538 | 166 |
 
**Observations**
 
- **Duplicates in train.** Train holds about 3 Roboflow copies per original photo (6568 images from 2140 originals). Valid and test hold one image per original.
- **Test boxes per class after tiling:** Hole 105, Seam 52, Stitch 172.
- **Median box area as % of the full image (train / test):**
  - Hole: 1.11 / 0.66
  - Seam: 15.93 / 7.43
  - Stitch: 9.47 / 5.66
- **Tiny holes.** Hole's 10th percentile is about 0.1–0.2% of the image, which is only a few pixels across.
- **Defect share vs. production.** About 54–61% of tiles contain a defect. The production base rate assumed later is 5%.

**Holes are much smaller than the other classes.** Hole boxes are smaller than Seam and Stitch boxes by a large margin. This suggests the model may have a harder time spotting them.
 
**The test split does not reflect a real line.** On a production line far fewer products are defective. The defective share here is deliberately high so there are enough defects to measure the model's ability to detect them. As a result, the raw test-set cost and false-alarm counts cannot be read as production figures, which is why the cost analysis reweights the test error rates to an assumed 5% production base rate.
 
**Seam size gap between train and test.** Train Seam boxes are about twice the size of test ones. This may affect the results, but it also acts as a test of the model: smaller seams are harder to detect, so detecting them successfully is stronger evidence that the model generalises.
 
---
 
## Finetuning and baselines
 
**Setup**
 
- **Task:** object detection.
- **Model:** YOLO11s, finetuned from COCO-pretrained weights. Ultralytics 8.4.191.
- **Training settings:**
  - 100 epochs (early stopping with patience 20 never triggered)
  - imgsz 320 (tiles at native resolution)
  - batch 16
  - seed 42
- **Hardware:** NVIDIA RTX 5070 Laptop GPU, PyTorch 2.11.0 with the CUDA 12.8 build. Trained locally.
- **Validation:** best mAP50-95 was 0.279, at epoch 94.
- **Test:** mAP50-95 is 0.191.
- **Weights:** `weights/fabric_defect_yolo11s.pt`.

**Results on the test split (550 tiles)**
 
| model | class-agnostic AP50 | mAP50-95 (3 classes) | tile ROC AUC |
|---|---|---|---|
| finetuned YOLO11s | 0.417 | 0.191 | 0.901 |
| COCO YOLO11s zero-shot | 0.023 | n/a (no fabric classes) | 0.628 |
| predict nothing | 0.000 | 0.000 | 0.500 |
 
Class-agnostic AP50 merges the three classes into one. This lets the COCO model, which has no fabric classes, be compared on the same footing.
 
**What the COCO baseline shows.** ROC AUC measures how often a randomly chosen defect tile receives a higher score than a randomly chosen clean tile. 0.5 is a coin flip and 1.0 is perfect. The COCO model reaches 0.628, only somewhat better than chance. It has never been trained on fabric defects, so it is most likely reacting to object-like structure in the tile, such as the visible lines of seams and stitches, rather than recognising defects. Its class-agnostic AP50 of 0.023 confirms that it cannot actually locate them. The finetuned model's AUC of 0.901 shows how much the finetuning added.
 
**Why validation (0.279) is higher than test (0.191).** Three factors likely contribute:
 
1. **The test defects are smaller.** Median box sizes are smaller in test than in validation (Hole 0.66% vs 1.32%, Seam 7.43% vs 9.72% of the image). Smaller defects are harder to detect, so the test split is harder.
2. **The test set is small.** With 179 original images and only 52 Seam boxes, a handful of hits or misses moves the score noticeably.
3. **Validation is slightly optimistic.** The best epoch was chosen by its validation score, so that score flatters the model a little.

---
 
## Per-class evaluation and failure modes
 
**Per-class results on test**
 
| class | test boxes | precision | recall | AP50 | mAP50-95 |
|---|---|---|---|---|---|
| Hole | 105 | 0.449 | 0.257 | 0.246 | 0.113 |
| Seam | 52 | 0.692 | 0.635 | 0.713 | 0.305 |
| Stitch | 172 | 0.532 | 0.462 | 0.452 | 0.154 |
| all | 329 | 0.557 | 0.451 | 0.470 | 0.191 |
 
**Confusion matrix on test** (rows are predicted, columns are true)
 
| | true Hole | true Seam | true Stitch | true background |
|---|---|---|---|---|
| pred Hole | 68 | 0 | 5 | 5915 |
| pred Seam | 5 | 45 | 1 | 996 |
| pred Stitch | 11 | 4 | 154 | 8786 |
| pred background | 21 | 3 | 12 | 0 |
 
**Notes**
 
- **Background column.** It counts every low-confidence box (conf ≥ 0.001), so its size reflects the evaluation's confidence floor. It is not a count of real false alarms.
- **Class mix-ups are few.** The largest is Holes predicted as Stitch (11).
- **Missed holes.** 21 of the 105 test holes were missed entirely.
- **Figures:** `artifacts/figures/test_confusion_matrix*.png` and `test_BoxPR_curve.png`.

**Why Hole performs worst.** Holes are the smallest defects in the photos, so the model has the least visual information to work with and has a harder time finding them. This is reflected in Hole's recall of 0.257, the lowest of the three classes.

---
 
## Cost matrix and operating point
 
**Assumptions**
 
| item | value |
|---|---|
| FN cost | €5 (estimated real-world range €10–20; see above) |
| FP cost | €1 |
| TP / TN cost | €0 |
| review cost | €0.10 per tile |
| production base rate | 5% defective tiles |
 
**Method**
 
- **Reweighting.** Test error rates are reweighted to the 5% production base rate rather than averaged over test tiles, because the test split is 54% defective.
- **Validation vs. test.** All thresholds are chosen on the **validation** split and reported on the **test** split.

**Single threshold at minimum validation cost: 0.14**
 
| metric (test) | value |
|---|---|
| recall | 0.669 |
| false-alarm rate | 0.075 |
| expected cost | €153.83 per 1,000 tiles |
 
**Sweep:** `artifacts/cost_sweep.csv` and `artifacts/figures/cost_vs_threshold.png`.
 
**Which assumption matters most.** The threshold balances two weights per tile:
 
- **misses:** base rate × FN cost = 5% × €5 = **0.25**
- **false alarms:** 95% × €1 = **0.95**
When misses become relatively heavier, the cheapest threshold moves lower and the model flags more tiles. Because base rate and FN cost multiply together, both are equally influential. FN cost is the assumption most likely to be wrong, however, since the estimated real-world cost of a defect reaching a customer is €10–20, not €5. At €15, misses would weigh 0.75 instead of 0.25, three times more, so the threshold would drop and a larger share of tiles would go to review.
 
---
 
## What a person still has to check
 
**Band chosen on validation:** pass < 0.05 ≤ review < 0.51 ≤ fail.
 
**Expected outcomes per 1,000 production tiles** (test rates at a 5% base rate)
 
| | pass | review | fail |
|---|---|---|---|
| defective (50) | 7.4 | 31.9 | 10.6 |
| clean (950) | 736.8 | 209.4 | 3.7 |
 
**Review workload**
 
- **Review share:** 24% of tiles. At 50,000 tiles/day, that is about 12,000 tiles/day.
- **Staffing:** at 5 s per tile, that needs 2.10 reviewers on 8 h shifts.
- **Placeholders:** tiles/day, seconds per tile and shift length are all placeholders.

**Comparison of designs**
 
| design | € per 1,000 tiles | € per day | review share | reviewers |
|---|---|---|---|---|
| do nothing (ship everything) | 250.00 | 12,500 | 0 | 0 |
| person checks every tile | 100.00 | 5,000 | 1.00 | 8.68 |
| model alone (threshold 0.14) | 153.83 | 7,692 | 0 | 0 |
| model + review (0.05–0.51) | 65.04 | 3,252 | 0.24 | 2.10 |
 
**Is the review load realistic?** A review time of 5 s per tile is realistic, because a tile is only a small piece of the whole product. Two reviewers is a reasonable staffing level for a quality-control team, so the 2.10 reviewers required is achievable.
 
**Should the model be shipped?** In its current state the model does not perform well enough to be shipped as a standalone system: on its own it costs more than having a person check every tile. Its potential is clear, however, and the results show it is most useful when it works together with quality-control staff through the review band.
 
---
 
## Limitations & next steps
 
**No defect-free images.** The biggest limitation was that the dataset contained no untagged photos of non-defective fabric, so the model had no true negative examples. Cutting images into tiles was used to work around this, but the tile edges can split features such as holes and stitches, which creates further confusion for the model. Collecting images of genuinely defect-free fabric would address this directly.
 
**Dataset size and variety.** The dataset is limited in size, and the fabric is often similar in colour. A larger and more varied dataset would help the model generalise to the range of fabrics found on a real line.
 
**Cost assumptions.** The cost analysis uses FN = €5, while the estimated real cost of a defect reaching a customer is €10–20. Rerunning the cost sweep with the real-world range would give a more accurate threshold and review band.
 
---
 
## Generative AI disclosure
 
**Claude (claude.ai chat) was used for:**
 
- **Brainstorming:** suggesting candidate business problems for the assignment, and checking that the candidate datasets on Roboflow Universe were public and had enough classes.
- **Environment troubleshooting:** diagnosing why training ran on the CPU (a CPU-only PyTorch build), installing the CUDA 12.8 build, resolving a conflict between two virtual environments, and explaining why `uv run --no-sync` was needed.
- **Reading results:** helping interpret the training and evaluation output.
- **Explaining concepts** I then adopted in my own analysis: what ROC AUC means for the COCO baseline, why validation and test scores can differ, and how the base rate and FN cost weigh against each other in the threshold.
- **Report writing:** formatting this report and turning my own answers into prose.

**My own decisions:**
 
- Choosing the fabric-defect problem and the `fabric-defect-thesis` dataset.
- Who acts on the decision, and framing the model as a tool that assists QC workers.
- The cost reasoning: a real-world estimate of €10–20 per defect reaching a customer, and a false alarm as a ~€1 manual re-check.
- Keeping FN = €5 in the analysis rather than rerunning it.
- The judgments on the review load (5 s per tile, ~2 reviewers).
- The conclusion that the model should not be deployed without human review.
- The limitations identified.

**Claude Code was used for:**
 
- **Writing the code.** Claude Code wrote most of the code in this repo:
  - the Roboflow download
  - the tiling (`tiling.py`)
  - the cost functions (`costing.py`)
  - the training and evaluation pipeline (`train.ipynb`, plus the earlier smoke test `smoke_train.*`)
  - the baselines, threshold sweep, review band and plots
  - the Gradio dashboard (`app.ipynb`)
- **Debugging.** It diagnosed and worked around the crashes during evaluation on my laptop. That led to `results.py`, which reruns only the evaluation and costing steps from the trained weights, in smaller chunks. The reported results come from that run.
- **Technical design choices, proposed by Claude Code and agreed with me:**
  - cutting images into tiles to get clean fabric
  - dropping train images that leak into valid/test
  - YOLO11s at 320 px
  - choosing thresholds on validation and reporting them on test
  - reweighting test rates to a production base rate
  - the class-agnostic AP50 / ROC AUC comparison against the COCO baseline

  I reviewed each one and can explain it.
- **Report.** It drafted the first version of this report as factual bullet points and tables from the run output. The interpretations, cost figures, review-load judgments and conclusions are mine (see above).