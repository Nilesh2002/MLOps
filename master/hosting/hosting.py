from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOK"))
api.upload_folder(
    folder_path="master/deployment",     # the local folder containing your files
    repo_id="nilku/TourismPackagePrediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
