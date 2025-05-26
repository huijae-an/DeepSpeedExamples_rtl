from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "rtl-llm/origen-descriptions"
# FILENAME = "data.json"

#hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", local_dir="")
snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir="assets")

