# Face Recognition Analysis Tool
Several scripts for face ReID metric, including an Interactive Streamlit app for comparing model-predicted face labels against ground-truth (GT) annotations across multiple videos or clips. It aligns frames from multiple files, matches predictions to GT via IoU ≥ 0.5, applies a user-set similarity score threshold, and visualizes:

- Label counts (incl. wrong matches and unknown)

- A confusion matrix (GT vs. Pred) with rich per-cell annotations

- Similarity score histograms (raw counts & normalized densities)

## ✨ Features
- Multi-file ingestion & reindexing
    Concatenates multiple GT and prediction JSONs by reindexing frame IDs with offsets so sequences don’t collide.

- Robust matching
    Greedy 1–1 assignment per frame using IoU ≥ 0.5 between predicted and GT boxes.

- Threshold-aware evaluation
    Predictions with score < threshold are counted as Unknown.

- Clear metrics
    Per-label counts (including wrong matches and unknown)
    Confusion matrix with row/column share annotations
    Accuracy by Predicted total and by GT total

- Insightful score distributions
    Raw-count and normalized-density histograms for Correct, Wrong, and Unknown groups.

## 🧩 Expected Input Formats
Ground Truth (GT) JSON
Array of identities, each containing a faces array with per-frame boxes:
```json
[
  {
    "id": "person_001",
    "faces": [
      {
        "frame_id": 0,
        "name": "Alice",
        "bounding_box": {
          "top_left": {"x": 100, "y": 120},
          "bottom_right": {"x": 220, "y": 300}
        }
      }
    ]
  }
]
```
Notes:

Frame indexing: frame_id is 0-indexed within each GT file. The app automatically offsets subsequent files so frames remain unique when concatenated.

Label usage: The app lowercases name for consistent matching.

Predictions JSON
Array of frames, each with an image filename and a faces list:
```json
[
  {
    "image": "frame0001.png",
    "faces": [
      {
        "label": "alice",
        "score": 0.91,
        "bbox": {"x1": 12, "y1": 22, "x2": 59, "y2": 98}
      }
    ]
  }
]
```
Notes:

Frame indexing: The app extracts the frame index from image via frame(\d+) (e.g., frame0042.png → 41 internally, converted to 0-index). If the pattern is missing, it falls back to the array index.

BBox format: Predictions must provide {x1, y1, x2, y2}. (GT uses nested top_left/bottom_right and is converted internally.)

## Matching & Metrics (How It Works)
1. IoU Matching (≥ 0.5)
    - For each predicted face in a frame, find the unmatched GT box with the highest IoU ≥ 0.5. If none, it’s considered a prediction with no GT match.

1. Threshold Handling
    - If pred.score < threshold:
        - If it overlaps a GT: logged as GT → unknown (counted under Unknown).
        - If no GT overlap: logged as none → unknown.
    - If pred.score ≥ threshold:
        - If matched to GT:
            - Correct if pred.label == gt.label
            - Wrong otherwise (increments wrong matches)
        - If unmatched to GT: counted under pred.label with GT = none.

1. Outputs
    - Label Counts: Counts for each predicted label, plus wrong matches and unknown.
    - Confusion Matrix: GT (rows) vs Predicted (columns).
        - Excludes none/unknown from main GT row index for clarity; unknown/none are placed at the end of columns if present.
    - Per-cell annotations show:
        - Raw count
        - Share of the GT row (val / row_total)
        - Share of the Pred column (val / col_total)
    - Accuracies:
        - Acc (Pred) = diagonal sum / total predicted
        - Acc (GT) = diagonal sum / total GT

## 🚀 Quickstart
(for sscm_webUI_multi-files_show-GT-and-predicted-total_2.py)
1) Install
```bash
Copy
Edit
# Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install streamlit matplotlib seaborn pandas
```
2) Run
```bash
Copy
Edit
streamlit run sscm_webUI_multi-files_show-GT-and-predicted-total_2.py
```

3) Use the App
    1. Open the Streamlit URL printed in your terminal.
    1. In the sidebar, upload:
        - Multiple Prediction JSON files
        - Multiple Ground Truth JSON files
    1. Adjust the Score Threshold slider.
    1. Explore:
        - Label Counts bar chart
        - Confusion Matrix with per-cell details & overall accuracies
        - Score Histograms (Raw / Density)

## 📂 File Concatenation & Reindexing Details
- GT files:
    The tool tracks the max local frame_id in each GT file and offsets the next file’s frames by max_frame_id + 1.

- Prediction files:
    The tool finds the max local index via image names (e.g., frame0123.png → 122), then offsets the next file by max_local + 1.

This lets you analyze multiple clips as one continuous sequence.