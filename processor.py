# Assuming we are provided with review_id and text from crawler_app.reviews_* tables

import time
import logging
from typing import Any, Dict, List, Optional

from processor_utils import pop, push
from processor_config import (
    BATCH_SIZE,
    EMPTY_QUEUE_SLEEP_SECONDS,
    AIGT_REPO_ID,
    AIGT_SUBDIR_BY_LANG_JSON,
    AIGT_REVISION,
    AIGT_HF_TOKEN,
    AIGT_DEVICE,
    AIGT_CACHE_POLICY,
    AIGT_MAX_LEN,
    AIGT_BATCH_SIZE,
    AIGT_WINDOW_AI_THRESHOLD,
    AIGT_PREFER_BF16,
)

from core import TextClassifier, TextInferConfig


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

classifier = None


def _init_classifier_if_needed():
    global classifier

    if classifier is not None:
        return

    logger.info("Initializing review text classifier.")

    cfg = TextInferConfig(
        repo_id = AIGT_REPO_ID,
        subdir_by_lang_json = AIGT_SUBDIR_BY_LANG_JSON,
        revision = AIGT_REVISION,
        hf_token = AIGT_HF_TOKEN,
        device = AIGT_DEVICE,
        cache_policy = AIGT_CACHE_POLICY,
        max_len = AIGT_MAX_LEN,
        batch_size = AIGT_BATCH_SIZE,
        window_ai_threshold = AIGT_WINDOW_AI_THRESHOLD,
        prefer_bf16 = AIGT_PREFER_BF16,
    )

    classifier = TextClassifier(cfg = cfg)
    classifier.load_models()

    logger.info("Review text classifier initialized successfully.")


def _coerce_lang(x: Any) -> str:
    if x is None:
        return "en"

    lang = str(x).lower().strip()

    if not lang or lang in {"nan", "none", "null"}:
        return "en"

    lang = lang.split("-")[0].split("_")[0]

    return lang or "en"


def _get_review_id(entry: Dict[str, Any], fallback: str) -> str:
    return str(entry.get("review_id") or fallback)


def _get_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("text") or "").strip()


def _get_lang(entry: Dict[str, Any]) -> str:
    return _coerce_lang(
        entry.get("lang")
        or entry.get("language")
        or entry.get("language_code")
    )


def _empty_prediction(
    *,
    review_id: str,
    lang: str,
    status: str = "empty_or_failed",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "review_id": review_id,
        "prediction_id": review_id,
        "lang": lang,
        "prediction_short": None,
        "prediction_long": None,
        "fraction_ai": None,
        "ai_probability": None,
        "human_probability": None,
        "n_windows": 0,
        "n_ai_segments": 0,
        "n_human_segments": 0,
        "n_tokens": 0,
        "error": error,
    }


def _build_results(
    *,
    entries: List[Dict[str, Any]],
    review_ids: List[str],
    langs: List[str],
    preds: List[Dict[str, Any]],
    job_id: str,
) -> List[Dict[str, Any]]:
    preds_by_id: Dict[str, Dict[str, Any]] = {}

    for pred in preds:
        pid = str(pred.get("prediction_id") or pred.get("review_id") or "")
        if pid and pid not in preds_by_id:
            preds_by_id[pid] = pred

    results: List[Dict[str, Any]] = []

    for i, entry in enumerate(entries):
        review_id = review_ids[i]
        lang = langs[i]

        pred = preds_by_id.get(review_id)

        if pred is None:
            pred = _empty_prediction(
                review_id = review_id,
                lang = lang,
                error = "missing_prediction",
            )

        results.append(
            {
                "job_id": job_id,
                "review_id": review_id,
                "prediction_id": review_id,
                "text": entry.get("text"),
                "ai_probability": pred.get("ai_probability"),
                "fraction_ai": pred.get("fraction_ai"),
                "n_tokens": pred.get("n_tokens"),

                # kept available for future debugging / schema expansion
                # "lang": pred.get("lang") or lang,
                # "prediction_short": pred.get("prediction_short"),
                # "prediction_long": pred.get("prediction_long"),
                # "human_probability": pred.get("human_probability"),
                # "n_windows": pred.get("n_windows"),
                # "n_ai_segments": pred.get("n_ai_segments"),
                # "n_human_segments": pred.get("n_human_segments"),
                # "status": pred.get("status"),
                # "error": pred.get("error"),
                # "bundle_id": entry.get("bundle_id"),
                # "store": entry.get("store"),
                # "score": entry.get("score"),
                # "crawl_date": entry.get("crawl_date"),
                # "timestamp": entry.get("timestamp"),
            }
        )

    return results


def _push_sub_batch(
    *,
    results: List[Dict[str, Any]],
    job_id: str,
    job_token: Optional[str],
    ack: bool,
):
    jobs_payload = [{"id": job_id, "token": job_token}] if ack else []

    processed_jobs = [
        {
            "jobs": jobs_payload,
            "filename": f"review_text_results_{job_id}_{int(time.time())}.json",
            "results": results,
        }
    ]

    push(processed_jobs)


def _extract_entries(job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = (
        job_data.get("jobs")
        or job_data.get("entries")
        or job_data.get("reviews")
    )

    if entries is None:
        entries = [job_data]

    if not isinstance(entries, list):
        raise ValueError("Expected job data to contain a list under jobs, entries, or reviews.")

    return entries


def process_batch():
    _init_classifier_if_needed()

    jobs = pop(batch_size = 1)

    if len(jobs) == 0:
        logger.info("No jobs received from queue. Sleeping.")
        time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)
        return

    job = jobs[0]
    job_id = str(job["id"])
    job_token = job.get("token")
    job_data = job.get("data") or {}

    entries = _extract_entries(job_data)
    total = len(entries)

    logger.info(f"Processing review job {job_id}: {total} entries")

    if total == 0:
        logger.warning(f"Job {job_id} has 0 entries. Acking immediately.")
        _push_sub_batch(
            results = [],
            job_id = job_id,
            job_token = job_token,
            ack = True,
        )
        return

    total_pushed = 0

    for chunk_start in range(0, total, BATCH_SIZE):
        chunk = entries[chunk_start : chunk_start + BATCH_SIZE]

        review_ids = [
            _get_review_id(
                entry,
                fallback = f"{job_id}_{chunk_start + i}",
            )
            for i, entry in enumerate(chunk)
        ]

        texts = [_get_text(entry) for entry in chunk]
        langs = [_get_lang(entry) for entry in chunk]

        n_empty_texts = sum(1 for text in texts if not text)

        logger.info(
            f"Sub-batch {chunk_start // BATCH_SIZE + 1}: "
            f"{len(texts)} reviews, empty_texts={n_empty_texts}"
        )

        preds = classifier.classify_texts_batch(
            texts = texts,
            langs = langs,
            prediction_ids = review_ids,
        )

        results = _build_results(
            entries = chunk,
            review_ids = review_ids,
            langs = langs,
            preds = preds,
            job_id = job_id,
        )

        is_last_chunk = (chunk_start + BATCH_SIZE) >= total

        _push_sub_batch(
            results = results,
            job_id = job_id,
            job_token = job_token,
            ack = is_last_chunk,
        )

        total_pushed += len(results)

    logger.info(f"Review job {job_id} complete: pushed {total_pushed} results")


def main():
    logger.info("Starting review text classification processor.")

    while True:
        try:
            process_batch()
        except Exception as e:
            logger.exception(f"Error in process_batch: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
