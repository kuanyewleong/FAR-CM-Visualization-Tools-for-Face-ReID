def calculate_far(pred_labels, gt_labels, unknown_label="unknown"):
    """
    Calculate False Acceptance Rate (FAR) considering 'unknown' as a valid rejection.

    Parameters:
    - pred_labels: list of predicted identity labels (or 'unknown')
    - gt_labels: list of ground truth identity labels
    - unknown_label: label representing unknown/no-match predictions

    Returns:
    - FAR as a rate (float)
    """
    if len(pred_labels) != len(gt_labels):
        raise ValueError("Prediction and ground truth lists must be the same length.")

    false_accepts = 0
    valid_attempts = 0

    for pred, gt in zip(pred_labels, gt_labels):
        if pred == unknown_label:
            continue  # Not a match attempt
        valid_attempts += 1
        if pred != gt:
            false_accepts += 1

    far = (false_accepts / valid_attempts) if valid_attempts > 0 else 0.0
    return far


preds = ["id1", "id1", "unknown", "id4", "id3"]
gts   = ["id1", "id3", "id2",     "id4", "id6"]

far = calculate_far(preds, gts)
print(f"FAR: {far:.2f}")
