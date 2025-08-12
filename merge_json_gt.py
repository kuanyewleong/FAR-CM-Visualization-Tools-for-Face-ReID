import json
import glob

def merge_json_with_incremental_frame_ids(file_paths):
    merged_data = []
    frame_offset = 0

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            updated_faces = []
            for face in entry['faces']:
                face['frame_id'] += frame_offset
                updated_faces.append(face)

            merged_data.append({
                "camera_name": entry["camera_name"],
                "video_path": entry["video_path"],
                "faces": updated_faces
            })

        # Update offset: assume last face's frame_id is the max in that file
        max_frame = max(face['frame_id'] for entry in data for face in entry['faces'])
        frame_offset = max_frame + 1

    return merged_data

# Example usage:
file_paths = sorted(glob.glob('ground_truth/5pc/*.json'))  # Modify path pattern as needed
merged = merge_json_with_incremental_frame_ids(file_paths)

# Save to a new file
with open('merged_output_5pc_0-10.json', 'w') as out_f:
    json.dump(merged, out_f, indent=4)
