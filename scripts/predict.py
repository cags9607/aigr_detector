# scripts/predict.py

from __future__ import annotations

import argparse
import os

import pandas as pd

from aigt import BatchConfig, Detector, WindowConfig


def main():
    ap = argparse.ArgumentParser(description = "AIGT review batch prediction")

    ap.add_argument("--input", "-i", required = True, help = "Input dataframe: parquet or csv")
    ap.add_argument("--text-col", default = "text")
    ap.add_argument("--id-col", default = "prediction_id")
    ap.add_argument("--lang-col", default = None)
    ap.add_argument("--output-prefix", "-o", default = "aigt_reviews_out")

    ap.add_argument(
        "--token",
        default = None,
        help = "HF token, or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var",
    )

    ap.add_argument(
        "--repo-id",
        default = "DeepSee-io/qwen_adapters_aigt",
    )

    ap.add_argument(
        "--token-length",
        type = int,
        default = 512,
    )

    ap.add_argument(
        "--batch-size",
        type = int,
        default = 16,
    )

    ap.add_argument(
        "--window-ai-threshold",
        type = float,
        default = 0.5,
    )

    args = ap.parse_args()

    # ---- load dataframe ----
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        raise ValueError("Unsupported input format. Use parquet or csv.")

    if args.text_col not in df.columns:
        raise ValueError(f"text column not found: {args.text_col}")

    if args.id_col not in df.columns:
        df[args.id_col] = range(len(df))

    texts = df[args.text_col].fillna("").astype(str).tolist()
    doc_ids = df[args.id_col].astype(str).tolist()

    if args.lang_col and args.lang_col in df.columns:
        langs = (
            df[args.lang_col]
            .fillna("en")
            .astype(str)
            .str.lower()
            .str.strip()
            .str.split("-", n = 1)
            .str[0]
            .str.split("_", n = 1)
            .str[0]
            .tolist()
        )
    else:
        langs = "en"

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    # ---- detector ----
    det = Detector.from_hf(
        repo_id = args.repo_id,
        subdir_by_lang = {
            "en": "reviews/best",
        },
        token = token,
    )

    articles_df, windows_df = det.predict(
        texts = texts,
        doc_ids = doc_ids,
        lang = langs,
        window = WindowConfig(token_length = int(args.token_length)),
        batch = BatchConfig(
            batch_size = int(args.batch_size),
            show_progress = True,
        ),
        window_ai_threshold = float(args.window_ai_threshold),
    )

    # ---- save outputs ----
    articles_path = f"{args.output_prefix}_articles.parquet"
    windows_path = f"{args.output_prefix}_windows.parquet"

    articles_df.to_parquet(articles_path, index = False)
    windows_df.to_parquet(windows_path, index = False)

    print("[OK] Saved:")
    print(" ", articles_path)
    print(" ", windows_path)


if __name__ == "__main__":
    main()
