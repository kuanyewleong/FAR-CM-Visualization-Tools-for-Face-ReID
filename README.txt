Face Recognition Analysis Tool
Several scripts for face ReID metric, including an Interactive Streamlit app for comparing model-predicted face labels against ground-truth (GT) annotations across multiple videos or clips. It aligns frames from multiple files, matches predictions to GT via IoU ≥ 0.5, applies a user-set similarity score threshold, and visualizes:

- Label counts (incl. wrong matches and unknown)

- A confusion matrix (GT vs. Pred) with rich per-cell annotations

- Similarity score histograms (raw counts & normalized densities)

✨ Features
Multi-file ingestion & reindexing
Concatenates multiple GT and prediction JSONs by reindexing frame IDs with offsets so sequences don’t collide.

Robust matching
Greedy 1–1 assignment per frame using IoU ≥ 0.5 between predicted and GT boxes.

Threshold-aware evaluation
Predictions with score < threshold are counted as Unknown.

Clear metrics

Per-label counts (including wrong matches and unknown)

Confusion matrix with row/column share annotations

Accuracy by Predicted total and by GT total

Insightful score distributions
Raw-count and normalized-density histograms for Correct, Wrong, and Unknown groups.

🧩 Expected Input Formats
Ground Truth (GT) JSON
Array of identities, each containing a faces array with per-frame boxes: