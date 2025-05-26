import json

def rename_key_in_json(input_file, output_file):
    # Load the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Check that it's a list of dictionaries
    if not isinstance(data, list):
        raise ValueError("Expected a list of dictionaries")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("List must contain dictionaries")
        if 'chosen' in item:
            item['response'] = item.pop('chosen')

    # Save the updated data back to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage
rename_key_in_json('train.json', 'train.json')
rename_key_in_json('eval.json', 'eval.json')
