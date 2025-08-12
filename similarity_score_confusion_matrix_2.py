import json
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd

# ---------- Bounding box IoU ----------
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

# ---------- Load data ----------
@st.cache_data
def load_data():
    with open("recognition_results_vid0.json") as f:
        pred_data = json.load(f)
    with open("face_data.json") as f:
        gt_raw = json.load(f)

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

# ---------- Counting + Confusion matrix ----------
def analyze(pred_data, gt_data, threshold):
    counter = Counter()
    confusion = []

    for i, frame in enumerate(pred_data):
        pred_faces = frame["faces"]
        gt_faces = gt_data[i]
        gt_matched = [False] * len(gt_faces)

        for pred in pred_faces:
            score = pred["score"]
            pred_label = pred["label"].lower()
            pred_bbox = pred["bbox"]

            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_faces):
                iou_val = iou(pred_bbox, gt["bbox"])
                if iou_val >= 0.5 and not gt_matched[idx] and iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = idx

            if score < threshold:
                if best_gt_idx != -1:
                    gt_label = gt_faces[best_gt_idx]["label"]
                    confusion.append((gt_label, "unknown"))
                    gt_matched[best_gt_idx] = True
                else:
                    confusion.append(("none", "unknown"))
                counter["unknown"] += 1
                continue

            if best_gt_idx != -1:
                gt_label = gt_faces[best_gt_idx]["label"]
                gt_matched[best_gt_idx] = True
                if pred_label == gt_label:
                    counter[pred_label] += 1
                else:
                    counter["wrong matches"] += 1
                confusion.append((gt_label, pred_label))
            else:
                counter[pred_label] += 1
                confusion.append(("none", pred_label))

    return counter, confusion

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("Face Recognition Evaluation Tool")

threshold = st.slider("Score Threshold", 0.0, 1.0, 0.5, 0.01)

label_counts, confusion_pairs = analyze(pred_data, gt_data, threshold)

# ---------- Layout ----------
col1, col2 = st.columns([2, 2])

# Bar chart
with col1:
    st.subheader("Label Counts")
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Threshold: {threshold:.2f}")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha='center', va='bottom')
    st.pyplot(fig)

# Confusion matrix
# Confusion matrix
with col2:
    st.subheader("Confusion Matrix (GT vs Predicted)")

    df = pd.DataFrame(confusion_pairs, columns=["Ground Truth", "Predicted"])

    # Determine unique labels
    gt_labels = sorted(set(df["Ground Truth"]))
    pred_labels = sorted(set(df["Predicted"]))

    # Move 'none' to bottom row (GT) and 'unknown' to last column (Pred)
    if "none" in gt_labels:
        gt_labels.remove("none")
        gt_labels.append("none")
    if "unknown" in pred_labels:
        pred_labels.remove("unknown")
        pred_labels.append("unknown")

    confusion_matrix = pd.crosstab(
        df["Ground Truth"], df["Predicted"],
        rownames=["GT"], colnames=["Predicted"],
        dropna=False
    ).reindex(index=gt_labels, columns=pred_labels, fill_value=0)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2)
    ax.set_title("GT vs Predicted")
    st.pyplot(fig2)

