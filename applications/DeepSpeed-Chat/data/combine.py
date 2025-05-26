import json
import random
import glob

def combine_and_shuffle_json(input_pattern, output_file):
    combined_data = []

    # Find all files matching the pattern
    for filename in glob.glob(input_pattern):
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                raise ValueError(f"File {filename} does not contain a list")

    # Shuffle the combined list
    random.shuffle(combined_data)

    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

# Example usage:
# Combines all JSON files in the current directory starting with 'data' and ending in .json
combine_and_shuffle_json('assets/*.json', 'train.json')
