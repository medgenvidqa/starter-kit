import torch
import numpy as np
from util import save_json, load_json
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import re, json, random
import numpy as np
import argparse
import json
import logging
from torch.nn.functional import sigmoid
from typing import Any, Dict, List, Optional, Union

import os
import re
import sys
import requests
import time
from datetime import datetime

from tqdm.auto import tqdm
def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_th_config(2026)
device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)

def setup_logging(log_file: str | None = None):
    level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str):
    return re.sub(r"[^\w.\-]+", "_", str(name)).strip("_") or "video"

def infer_video_extension(url: str):
    base = url.split("?")[0].split("#")[0]
    ext = os.path.splitext(base)[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}:
        return ext
    return ".mp4"
def infer_video_extension(url: str):
    base = url.split("?")[0].split("#")[0]
    ext = os.path.splitext(base)[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}:
        return ext
    return ".mp4"
def get_local_video_path(video_url: str, topic_id: str, video_dir: str):
    ext = infer_video_extension(video_url)
    return os.path.join(video_dir, safe_filename(topic_id) + ext)
def download_file(url: str, output_path: str, timeout: int = 60, chunk_size: int = 1024 * 1024) :
    ensure_dir(os.path.dirname(output_path))

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("Using cached video: %s", output_path)
        return output_path

    tmp_path = output_path + ".part"
    logger.info("Downloading video: %s", url)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    os.replace(tmp_path, output_path)
    logger.info("Saved video to: %s", output_path)
    return output_path

def seconds_to_mmss(seconds: Optional[float]) :
    if seconds is None:
        return None
    total = int(round(seconds))
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


def hhmmss_to_seconds(ts: str):
    ts = ts.strip()
    parts = ts.split(":")
    try:
        parts_i = [int(p) for p in parts]
    except ValueError:
        return None

    if len(parts_i) == 2:
        return parts_i[0] * 60 + parts_i[1]
    if len(parts_i) == 3:
        return parts_i[0] * 3600 + parts_i[1] * 60 + parts_i[2]
    return None

def parse_timelens_output(answer: str) :

    if not answer:
        return None, None

    text = answer.strip()
    text = re.sub(r"[–—]", "-", text)

    m = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)", text)
    if m:
        start = hhmmss_to_seconds(m.group(1))
        end = hhmmss_to_seconds(m.group(2))
        return start, end

    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    ms = re.search(r"start\s*[:=]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    me = re.search(r"end\s*[:=]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if ms and me:
        return float(ms.group(1)), float(me.group(1))

    return None, None

def build_messages(video_path: str, query: str, fps: float = 2.0):
    prompt = (
        f"You are given a full video and the question: '{query}'. "
        "Locate the single continuous video segment in the video that best answers the question. "
        "Return exactly one start time and one end time for the most relevant segment. "
        "Choose the shortest segment that contains enough evidence to answer the question. "
        "If multiple candidate segments exist, return only the best one. "
        "If the video does not contain an answer, return 'none'. "
        "Output exactly in this format: 'The answer segment is <start time> - <end time> seconds' or 'none'."
    )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 64 * 28 * 28,
                    "total_pixels": 14336 * 28 * 28,
                    "fps": fps,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

def run_timelens_on_video(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    video_path: str,
    query: str,
    fps: float = 2.0,
    max_new_tokens: int = 128,
):
    messages = build_messages(video_path=video_path, query=query, fps=fps)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images, videos = process_vision_info(messages, return_video_metadata=True)

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    start_sec, end_sec = parse_timelens_output(answer)

    return {
        "raw_answer": answer,
        "predicted_start_sec": start_sec,
        "predicted_end_sec": end_sec,
        "predicted_start_mmss": seconds_to_mmss(start_sec),
        "predicted_end_mmss": seconds_to_mmss(end_sec),
    }
def get_baseline_results(path_to_topics, model, processor, path_to_download_videos, path_to_results):

    data = load_json(path_to_topics)
    results = []
    for item in tqdm(data):
        topic_id = str(item['id'])
        query = item['question']
        video_url = item["video"]
        logger.info(f"Processing Question ID {topic_id}...")
        try:
            local_video_path = get_local_video_path(video_url, topic_id, path_to_download_videos)
            download_file(video_url, local_video_path)

            pred = run_timelens_on_video(
                model=model,
                processor=processor,
                video_path=local_video_path,
                query=query,
                fps=2
            )

            results.append(
                {
                    "id": topic_id,
                    "answer_start": pred["predicted_start_mmss"],
                    "answer_end": pred["predicted_end_mmss"]
                }
            )

        except Exception as e:
            logger.exception("Failed processing Question ID %s: %s", topic_id, repr(e))
            results.append(
                {
                    "id": topic_id,
                    "answer_start": None,
                    "answer_end": None
                }
            )

    save_json(results, path_to_results)
    logger.info("Results saved to %s", path_to_results)
    return results


def load_timelens_model(
    model_name,
    use_flash_attention: bool = False,
):

    model_kwargs = {
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
    }

    if use_flash_attention and device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **model_kwargs,
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )

    if device == "cpu":
        model.to(device)

    model.eval()
    return model, processor


def main() :
    parser = argparse.ArgumentParser(description="task-b baseline")
    parser.add_argument("--path_to_topics",  default="../data/TestDatasets/task_c_test.json")
    parser.add_argument("--path_to_download_videos", default="../data/Videos")
    parser.add_argument("--model_path_or_name", default="TencentARC/TimeLens-7B")
    parser.add_argument("--path_to_save_results", default="../data/BaselineResults")



    args = parser.parse_args()
    log_name = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file=log_name)
    ensure_dir(args.path_to_save_results)
    ensure_dir(args.path_to_download_videos)
    model, processor = load_timelens_model(
        model_name=args.model_path_or_name,
        use_flash_attention=False,
    )
    get_baseline_results(args.path_to_topics, model, processor, args.path_to_download_videos,
                                     path_to_results=os.path.join(args.path_to_save_results, "task_c_baseline_results.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())