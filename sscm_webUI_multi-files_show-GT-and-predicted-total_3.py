import json
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
import re

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

# ---------- Load multiple JSON files with reindexing ----------
def load_multiple_json_files(pred_files, gt_files):
    pred_data_all = []
    gt_data = defaultdict(list)
    frame_offset = 0

    # Load GT files with frame reindexing
    for file in gt_files:
        try:
            gt_raw = json.load(file)
            max_frame_id = 0
            for face in gt_raw:
                for f in face["faces"]:
                    local_frame_id = f["frame_id"]
                    new_frame_id = local_frame_id + frame_offset
                    name = f["name"].lower()
                    bbox = {
                        "x1": f["bounding_box"]["top_left"]["x"],
                        "y1": f["bounding_box"]["top_left"]["y"],
                        "x2": f["bounding_box"]["bottom_right"]["x"],
                        "y2": f["bounding_box"]["bottom_right"]["y"],
                    }
                    gt_data[new_frame_id].append({"label": name, "bbox": bbox})
                    max_frame_id = max(max_frame_id, local_frame_id)
            frame_offset += max_frame_id + 1
        except Exception as e:
            st.error(f"Error reading GT file {file.name}: {e}")

    # Reset for prediction frame offsetting
    frame_offset = 0
    for file in pred_files:
        try:
            pred_raw = json.load(file)
            for i, frame in enumerate(pred_raw):
                # Extract numeric frame index from filename (e.g., frame0042.png -> 42)
                match = re.search(r'frame(\d+)', frame["image"])
                if match:
                    local_frame_id = int(match.group(1)) - 1  # Convert to 0-indexed
                else:
                    local_frame_id = i  # fallback

                new_frame_id = local_frame_id + frame_offset
                pred_data_all.append({
                    "frame_id": new_frame_id,
                    "faces": frame["faces"]
                })

            max_local = max([int(re.search(r'frame(\d+)', f["image"]).group(1)) - 1
                             for f in pred_raw if re.search(r'frame(\d+)', f["image"])])
            frame_offset += max_local + 1
        except Exception as e:
            st.error(f"Error reading prediction file {file.name}: {e}")

    # Sort by global frame_id
    pred_data_all = sorted(pred_data_all, key=lambda x: x["frame_id"])
    return pred_data_all, gt_data

# ---------- Analyze ----------
def analyze(pred_data, gt_data, threshold):
    counter = Counter()
    confusion = []
    score_groups = {"Correct": [], "Wrong": [], "Unknown": []}

    for frame in pred_data:
        frame_id = frame["frame_id"]
        pred_faces = frame["faces"]
        gt_faces = gt_data[frame_id]
        gt_matched = [False] * len(gt_faces)

        for pred in pred_faces:
            score = pred["score"]
            pred_label = pred["label"].lower()
            pred_bbox = pred["bbox"]

            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_faces):
                if gt_matched[idx]:
                    continue
                iou_val = iou(pred_bbox, gt["bbox"])
                if iou_val >= 0.5 and iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = idx

            if score < threshold:
                if best_gt_idx != -1:
                    gt_label = gt_faces[best_gt_idx]["label"]
                    confusion.append((gt_label, "unknown"))
                    score_groups["Unknown"].append(score)
                    gt_matched[best_gt_idx] = True
                else:
                    confusion.append(("none", "unknown"))
                    score_groups["Unknown"].append(score)
                counter["unknown"] += 1
                continue

            if best_gt_idx != -1:
                gt_label = gt_faces[best_gt_idx]["label"]
                gt_matched[best_gt_idx] = True
                if pred_label == gt_label:
                    counter[pred_label] += 1
                    confusion.append((gt_label, pred_label))
                    score_groups["Correct"].append(score)
                else:
                    counter["wrong matches"] += 1
                    confusion.append((gt_label, pred_label))
                    score_groups["Wrong"].append(score)
            else:
                counter[pred_label] += 1
                confusion.append(("none", pred_label))
                score_groups["Unknown"].append(score)

    return counter, confusion, score_groups

def optimal_bins(data):
    n = len(data)
    return min(max(n // 2, 10), 100) if n > 0 else 10

# FAR calculation
def compute_far(confusion_pairs, unknown_label="unknown"):
    false_accepts = 0
    valid_attempts = 0
    for gt_label, pred_label in confusion_pairs:
        gt_label = gt_label.lower()
        pred_label = pred_label.lower()
        if pred_label != unknown_label:
            valid_attempts += 1
            if pred_label != gt_label:
                false_accepts += 1
    far = (false_accepts / valid_attempts) if valid_attempts > 0 else 0.0
    return far

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")
st.title("Face Recognition Analysis Tool")

# File upload UI
st.sidebar.header("Upload JSON Files")
pred_files = st.sidebar.file_uploader("Upload Prediction JSON Files", type="json", accept_multiple_files=True)
gt_files = st.sidebar.file_uploader("Upload Ground Truth JSON Files", type="json", accept_multiple_files=True)

threshold = st.slider("Score Threshold", 0.0, 1.0, 0.5, 0.01)

if pred_files and gt_files:
    pred_data, gt_data = load_multiple_json_files(pred_files, gt_files)
    label_counts, confusion_pairs, score_groups = analyze(pred_data, gt_data, threshold)

    # Bar chart
    col1, col2 = st.columns([2, 2])
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

    # Confusion Matrix & FAR
    with col2:
        st.subheader("Confusion Matrix (GT vs Predicted)")

        df = pd.DataFrame(confusion_pairs, columns=["Ground Truth", "Predicted"])
        gt_labels = sorted(set(df["Ground Truth"]) - {"none", "unknown"})
        pred_labels = sorted(set(df["Predicted"]))

        # Move 'unknown' and 'none' to the end for display clarity
        for special in ["unknown", "none"]:
            if special in pred_labels:
                pred_labels.remove(special)
                pred_labels.append(special)

        confusion_matrix = pd.crosstab(
            df["Ground Truth"], df["Predicted"],
            rownames=["GT"], colnames=["Predicted"],
            dropna=False
        ).reindex(index=gt_labels, columns=pred_labels, fill_value=0)

        # Metrics
        total_predicted = confusion_matrix.sum().sum()
        total_gt = confusion_matrix.sum(axis=1).sum()
        correct_predictions = sum(
            confusion_matrix.loc[label, label]
            for label in confusion_matrix.index
            if label in confusion_matrix.columns
        )

        accuracy_pred = correct_predictions / total_predicted if total_predicted > 0 else 0
        accuracy_gt = correct_predictions / total_gt if total_gt > 0 else 0
        far = compute_far(confusion_pairs, unknown_label="unknown")

        # Annotate the matrix
        annotations = confusion_matrix.copy().astype(str)
        row_totals = confusion_matrix.sum(axis=1)
        col_totals = confusion_matrix.sum(axis=0)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                val = confusion_matrix.iloc[i, j]
                gt_label = confusion_matrix.index[i]
                pred_label = confusion_matrix.columns[j]
                row_total = row_totals[gt_label]
                col_total = col_totals[pred_label]
                annotations.iloc[i, j] = f"{val}\n{val}/{row_total} (GT)\n{val}/{col_total} (Pred)"

        # Plot
        fig2, ax2 = plt.subplots(figsize=(9, 7))
        sns.heatmap(confusion_matrix, annot=annotations, fmt="", cmap="Blues", cbar=False, ax=ax2)
        ax2.set_title(
            f"GT vs Predicted | Acc (Pred): {accuracy_pred:.2%} | Acc (GT): {accuracy_gt:.2%} | FAR: {far:.2%}\n"
            f"Correct: {correct_predictions} / Pred: {total_predicted}, GT: {total_gt}"
        )
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("Ground Truth Label")
        st.pyplot(fig2)


    # Histograms
    st.subheader("Similarity Score Distribution: Raw Count vs Normalized Density")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Raw Count Histogram**")
        fig_raw, ax_raw = plt.subplots(figsize=(8, 4))
        has_data = False
        if score_groups["Correct"]:
            ax_raw.hist(score_groups["Correct"], bins=80, alpha=0.8, color='green', label='Correct Matches')
            has_data = True
        if score_groups["Wrong"]:
            ax_raw.hist(score_groups["Wrong"], bins=50, alpha=0.8, color='red', label='Wrong Matches')
            has_data = True
        if score_groups["Unknown"]:
            ax_raw.hist(score_groups["Unknown"], bins=50, alpha=0.5, color='gray', label='Unknown or Low Similarity')
            has_data = True
        if has_data:
            ax_raw.set_xlim(0.0, 1.0)
            ax_raw.set_xticks([round(x * 0.1, 1) for x in range(11)])
            ax_raw.set_xlabel("Cosine Similarity Score")
            ax_raw.set_ylabel("Count")
            ax_raw.legend()
            st.pyplot(fig_raw)
        else:
            st.info("No score data available for this threshold.")

    bins_correct = optimal_bins(score_groups["Correct"])
    bins_wrong = optimal_bins(score_groups["Wrong"])
    bins_unknown = optimal_bins(score_groups["Unknown"])

    with col4:
        st.markdown("**Normalized Density Histogram**")
        fig_density, ax_density = plt.subplots(figsize=(8, 4))
        has_density_data = False
        if score_groups["Correct"]:
            ax_density.hist(score_groups["Correct"], bins=bins_correct, alpha=0.8, color='green',
                            label='Correct Matches', density=True)
            has_density_data = True
        if score_groups["Wrong"]:
            ax_density.hist(score_groups["Wrong"], bins=bins_wrong, alpha=0.8, color='red',
                            label='Wrong Matches', density=True)
            has_density_data = True
        if score_groups["Unknown"]:
            ax_density.hist(score_groups["Unknown"], bins=bins_unknown, alpha=0.5, color='gray',
                            label='Unknown or Low Similarity', density=True)
            has_density_data = True
        if has_density_data:
            ax_density.set_xlim(0.0, 1.0)
            ax_density.set_xticks([round(x * 0.1, 1) for x in range(11)])
            ax_density.set_xlabel("Cosine Similarity Score")
            ax_density.set_ylabel("Density")
            ax_density.legend()
            st.pyplot(fig_density)
        else:
            st.info("No score data to display in normalized form.")
else:
    st.info("Please upload **multiple ground truth** and **prediction JSON files** to begin.")
