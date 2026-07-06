from huggingface_hub import HfApi

# Config
REPO_ID = "ankitbelbase034/experiments_checkpoints"
FOLDER_PATH = "A:/vlm_kvasir_full_continued/latest"  # Update this if you upload it to Colab's local space first
HF_TOKEN = ""

api = HfApi()

print("Starting upload... Colab's network will make this much faster!")

api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    path_in_repo="latest",
    repo_type="model",
    token=HF_TOKEN,
    
)

print("Upload complete!")