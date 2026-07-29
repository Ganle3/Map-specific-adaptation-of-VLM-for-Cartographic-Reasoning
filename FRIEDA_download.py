from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="knowledge-computing/FRIEDA",
    repo_type="dataset",
    local_dir=r"C:\Users\junyhuang\Thesis\Map-specific-adaptation-of-VLM-for-Cartographic-Reasoning\Datasets\FRIEDA",
)

print("FRIEDA完整仓库下载完成。")