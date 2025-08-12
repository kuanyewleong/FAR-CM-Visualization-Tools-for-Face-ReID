import json

# Load the original JSON
with open('gt_5pc_vid10.json', 'r') as f:
    data = json.load(f)

# Initialize containers
left_entries = []
right_entries = []

# Case 1: data is a list of dicts
if isinstance(data, list):
    for item in data:
        if item.get("camera_name") == "left":
            left_entries.append(item)
        elif item.get("camera_name") == "right":
            right_entries.append(item)

# Case 2: data is a dict containing a list under a key
elif isinstance(data, dict):
    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if item.get("camera_name") == "left":
                        left_entries.append(item)
                    elif item.get("camera_name") == "right":
                        right_entries.append(item)
            break  # Assumes only one such list is relevant

# Save to new JSON files
with open('gt_5pc_vid10_left.json', 'w') as f_left, open('gt_5pc_vid10_right.json', 'w') as f_right:
    json.dump(left_entries, f_left, indent=4)
    json.dump(right_entries, f_right, indent=4)
