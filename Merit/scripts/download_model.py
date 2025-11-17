from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repo_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
local_dir = "/hdd/intern/MERIT/llama-3.1-8b-instruct"

logger.info(f"Starting download of model '{repo_id}' to '{local_dir}'...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False # Windows가 아닌 환경에서는 False가 안정적입니다.
)
logger.info("Model download complete.")
