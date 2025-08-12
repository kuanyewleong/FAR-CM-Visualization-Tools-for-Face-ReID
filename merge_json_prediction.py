import json
import re
from glob import glob

def extract_frame_number(filename):
    match = re.search(r'frame(\d+)\.png', filename)
    return int(match.group(1)) if match else None

def merge_json_files_sequentially(file_paths):
    merged_data = []
    frame_counter = 1

    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)

        for frame in data:
            new_frame_name = f"frame{frame_counter:04d}.png"
            frame['image'] = new_frame_name
            merged_data.append(frame)
            frame_counter += 1

    return merged_data

# Example usage
input_files = sorted(glob("prediction/*.json"))  # or manually list the paths
merged = merge_json_files_sequentially(input_files)

# Save to output
with open('merged_results_5pc_0-10.json', 'w') as out_file:
    json.dump(merged, out_file, indent=4)
