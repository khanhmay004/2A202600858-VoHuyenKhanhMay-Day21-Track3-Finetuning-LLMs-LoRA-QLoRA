from huggingface_hub import create_repo, upload_folder
repo_id = "maykhanh004/qwen2.5-3b-vi-lab21-r16"   # đổi <hf-username>
create_repo(repo_id, repo_type="model", exist_ok=True)
upload_folder(
    repo_id=repo_id,
    folder_path="lab21_lora_t4/r16",
    repo_type="model",
    ignore_patterns=["checkpoint-*", "*.bin"],   # bỏ checkpoint + training_args.bin
)
print("✓ https://huggingface.co/" + repo_id)
