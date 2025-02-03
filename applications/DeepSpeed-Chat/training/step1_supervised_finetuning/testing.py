import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer


def upload_model_to_huggingface(model_name, repo):
    # Load the model and tokenizer from Hugging Face's model hub
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config, torch_dtype=torch.bfloat16)

    # Push the tokenizer and model to the specified Hugging Face repository
    print(f"Pushing tokenizer to Hugging Face repo: {repo}")
    tokenizer.push_to_hub(repo)

    print(f"Pushing model to Hugging Face repo: {repo}")
    model.push_to_hub(repo)


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Upload a Hugging Face model to your own repository.")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to fetch from Hugging Face (e.g., 'codellama/CodeLlama-7b')"
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository to upload the model and tokenizer to (e.g., username/repo-name)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Upload the model and tokenizer to your Hugging Face repository
    upload_model_to_huggingface(args.model_name, args.repo)


if __name__ == "__main__":
    main()
