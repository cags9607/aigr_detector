import io
import logging

import boto3
import pandas as pd
import requests
from botocore.config import Config

from processor_config import (
    QUEUE_API_KEY,
    QUEUE_URL,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    S3_ENDPOINT,
    S3_BUCKET,
    S3_FILE_PREFIX,
)

logger = logging.getLogger(__name__)

QUEUE_KEY = "AI_TEXT_V2"

S3_SESSION = boto3.session.Session(
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
)

S3_CLIENT = S3_SESSION.client(
    service_name="s3",
    endpoint_url=S3_ENDPOINT,
    config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
)


def read_parquet_columns(filename: str, columns: list) -> pd.DataFrame:
    key = f"{S3_FILE_PREFIX}{filename}"
    resp = S3_CLIENT.get_object(Bucket=S3_BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return pd.read_parquet(buf, columns=columns)


def pop(batch_size: int = 1) -> list:
    headers = {"x-api-key": QUEUE_API_KEY}
    data = {"key": QUEUE_KEY, "get": batch_size}

    resp = requests.post(url = QUEUE_URL, json = data, headers = headers)
    resp.raise_for_status()

    payload = resp.json()
    return payload["data"]["jobs"]


def push(processed_jobs: list):
    headers = {"x-api-key": QUEUE_API_KEY}
    data = {"key": QUEUE_KEY, "put": processed_jobs}

    resp = requests.post(url = QUEUE_URL, json = data, headers = headers)
    resp.raise_for_status()
