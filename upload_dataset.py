from huggingface_hub import HfApi

api = HfApi(token="")

# Em vez de upload_folder, use este para pastas grandes:
api.upload_large_folder(
    repo_id="jmn93/LIST",
    repo_type="dataset",
    folder_path=r"C:\Users\jeff_\Pictures\LIST"
)