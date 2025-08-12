import json
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ---------- Bounding box utilities ----------
def iou(boxA, boxB):
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    boxBArea = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])
    return interArea / float(boxAArea + boxBArea - interArea)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    with open("recognition_results_vid0.json") as f:
        pred_data = json.load(f)
    with open("face_data.json") as f:
        gt_raw = json.load(f)

    # Reorganize GT data: frame -> list of face entries
    gt_data = defaultdict(list)
    for face in gt_raw:
        for f in face["faces"]:
            frame_idx = f["frame_id"]
            name = f["name"].lower()
            bbox = {
                "x1": f["bounding_box"]["top_left"]["x"],
                "y1": f["bounding_box"]["top_left"]["y"],
                "x2": f["bounding_box"]["bottom_right"]["x"],
                "y2": f["bounding_box"]["bottom_right"]["y"],
            }
            gt_data[frame_idx].append({"label": name, "bbox": bbox})
    return pred_data, gt_data

pred_data, gt_data = load_data()

# ---------- Count logic ----------
def count_labels(pred_data, gt_data, threshold):
    counter = Counter()
    for i, frame in enumerate(pred_data):
        frame_name = frame["image"]
        pred_faces = frame["faces"]
        gt_faces = gt_data[i]

        matched_gt = set()

        for pred in pred_faces:
            score = pred["score"]
            pred_label = pred["label"].lower()
            pred_bbox = pred["bbox"]

            if score < threshold:
                counter["unknown"] += 1
                continue

            matched = False
            for gt in gt_faces:
                iou_val = iou(pred_bbox, gt["bbox"])
                if iou_val >= 0.5:
                    matched = True
                    matched_gt.add(tuple(gt["bbox"].items()))  # Just to track uniqueness
                    if pred_label == gt["label"]:
                        counter[pred_label] += 1
                    else:
                        counter["wrong matches"] += 1
                    break

            if not matched:
                # No match found in GT (or IoU too low) — could be an extra prediction
                counter[pred_label] += 1
    return counter

# ---------- Streamlit UI ----------
st.title("Face Label Comparison with Ground Truth")
threshold = st.slider("Score Threshold", 0.0, 1.0, 0.5, 0.01)

label_counts = count_labels(pred_data, gt_data, threshold)

# ---------- Visualization ----------
labels = list(label_counts.keys())
counts = list(label_counts.values())

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, counts)
ax.set_xlabel("Labels")
ax.set_ylabel("Count")
ax.set_title(f"Face Match Summary (Threshold: {threshold:.2f})")

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            str(count), ha='center', va='bottom')

st.pyplot(fig)
