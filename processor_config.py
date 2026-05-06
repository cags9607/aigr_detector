import os

QUEUE_API_KEY = os.getenv("QUEUE_API_KEY", "")
QUEUE_URL = os.getenv("QUEUE_URL", "https://deepsee-queue.herokuapp.com/exchange-batch")

# S3 – parquet file storage
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_FILE_PREFIX = os.getenv("S3_FILE_PREFIX", "")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
EMPTY_QUEUE_SLEEP_SECONDS = int(os.getenv("EMPTY_QUEUE_SLEEP_SECONDS", "60"))

# Text classifier: HF / LoRA
AIGT_REPO_ID = os.getenv("AIGT_REPO_ID", "DeepSee-io/qwen_adapters_aigt")
AIGT_SUBDIR_BY_LANG_JSON = os.getenv(
    "AIGT_SUBDIR_BY_LANG_JSON",
    '{"en":"reviews/best"}'
)
AIGT_REVISION = os.getenv("AIGT_REVISION", "") or None
AIGT_HF_TOKEN = os.getenv("AIGT_HF_TOKEN", "") or None

# Runtime knobs
AIGT_DEVICE = os.getenv("AIGT_DEVICE", "cuda")
AIGT_CACHE_POLICY = os.getenv("AIGT_CACHE_POLICY", "keep")
AIGT_MAX_LEN = int(os.getenv("AIGT_MAX_LEN", "500"))
AIGT_BATCH_SIZE = int(os.getenv("AIGT_BATCH_SIZE", "16"))
AIGT_WINDOW_AI_THRESHOLD = float(os.getenv("AIGT_WINDOW_AI_THRESHOLD", "0.5"))
AIGT_PREFER_BF16 = os.getenv("AIGT_PREFER_BF16", "1") not in ("0", "false", "False")
