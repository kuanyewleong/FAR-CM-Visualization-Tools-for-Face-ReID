import json
from pathlib import Path

def correct_frame_numbers(json_data):
    """Correct frame numbers to start from 'frame0001.png'."""
    for i, item in enumerate(json_data, start=1):
        new_frame_number = f"{i:04d}"
        item["image"] = f"frame{new_frame_number}.png"
    return json_data

def process_json_files(input_folder, output_folder=None, overwrite=False):
    """
    Process all JSON files in a folder to correct frame numbering.
    
    Args:
        input_folder (str): Path to the folder containing JSON files.
        output_folder (str, optional): If provided, saves corrected files here.
                                      If None, modifies files in-place (if overwrite=True).
        overwrite (bool): If True, overwrites original files (use with caution!).
    """
    input_folder = Path(input_folder)
    
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    elif not overwrite:
        raise ValueError("Either specify output_folder or set overwrite=True to modify files in-place.")
    
    for json_file in input_folder.glob("*.json"):
        # Read the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Correct frame numbers
        corrected_data = correct_frame_numbers(data)
        
        # Determine output path
        if output_folder:
            output_path = output_folder / json_file.name
        else:
            output_path = json_file
        
        # Save the corrected JSON
        with open(output_path, 'w') as f:
            json.dump(corrected_data, f, indent=4)
        
        print(f"Processed: {json_file} → {output_path}")

# Example usage:
input_dir = "analysis_results/faceme_output/VH11/5pe_right"
output_dir = None #"path/to/output_folder"  # Set to None if you want in-place modification (with overwrite=True)

process_json_files(input_dir, output_dir, overwrite=True)