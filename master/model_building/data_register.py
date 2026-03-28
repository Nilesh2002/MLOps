import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOK"))

repo_id = "nilku/Tourism-Packag-Prediction"
repo_type = "dataset" # Specify repo_type as dataset

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="master/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
