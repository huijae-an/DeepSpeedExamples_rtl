import argparse
import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dschat.utils.utils import load_hf_tokenizer
from dschat.utils.model.model_utils import create_hf_model


def validate_model_path(model_path):
    # List of required files in the model directory
    required_files = ["config.json", "pytorch_model.bin"]

    # Check if the directory exists
    if not os.path.isdir(model_path):
        print(f"Error: The path '{model_path}' is not a valid directory.")
        sys.exit(1)

    # Check for the presence of required files
    missing_files = [file for file in required_files if not os.path.isfile(os.path.join(model_path, file))]
    if missing_files:
        print(f"Error: The directory '{model_path}' is missing the following required files: {', '.join(missing_files)}")
        sys.exit(1)

    print(f"The path '{model_path}' is valid and contains all required files.")

def load_tokenizer_and_model(path, repo):
   

    # Fix fast_tokenizer back to True
    tokenizer = load_hf_tokenizer(path, fast_tokenizer=True)
    print("tokenizer loaded no problem")

    model = create_hf_model(AutoModelForCausalLM, path, tokenizer)
    print("model loaded no problem")


    tokenizer.push_to_hub(repo)
    model.push_to_hub(repo)



def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process model location path.")
    
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the model location"
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository to upload the model and tokenizer to (e.g., username/repo-name)"
    )    
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Validate the path
    validate_model_path(args.path)

    # Load Tokenizer and Model, and upload to Huggingface
    load_tokenizer_and_model(args.path, args.repo)

if __name__ == "__main__":
    main()
