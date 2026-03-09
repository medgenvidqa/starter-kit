from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from datetime import datetime
import requests

try:
    import ijson
except ImportError:
    ijson = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger()

_TS_LINE_RE = re.compile(
    r"""^\s*\[\s*            
        \d+(?:\.\d+)?\s*s    
        \s*-\s*
        \d+(?:\.\d+)?\s*s    
        \s*\]\s*$            
    """,
    re.VERBOSE
)

def strip_timestamps(transcript: str) -> str:

    if not transcript:
        return transcript

    out_lines = []
    for line in transcript.splitlines():
        line_stripped = line.strip()

        if _TS_LINE_RE.match(line_stripped):
            continue

        if line_stripped in {".", "•", "·"}:
            continue

        out_lines.append(line)

    cleaned = "\n".join(out_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()



def setup_logging(verbose: bool, log_file: str | None = None):
    level = logging.DEBUG if verbose else logging.INFO
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


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def load_cache_json(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Failed to load cache %s: %s", path, repr(e))
    return {}

def save_cache_json_atomic(path: str, cache: Dict[str, str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    os.replace(tmp, path)

def caption_segments_to_text(payload: Any):
    if not isinstance(payload, dict):
        return ""
    segs = payload.get("caption_segments")
    if segs is None:
        return ""
    if isinstance(segs, str):
        return normalize_whitespace(segs)

    parts: List[str] = []
    if isinstance(segs, list):
        for seg in segs:
            if seg is None:
                continue
            if isinstance(seg, str) and seg.strip():
                parts.append(seg.strip())
            elif isinstance(seg, dict):
                for k in ("text", "caption", "utf8", "raw", "content", "transcript"):
                    v = seg.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                        break
    return normalize_whitespace("\n".join(parts))

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
def build_new_subtitles_map_for_ids_streaming(
    subtitles_paths: List[str],
    wanted_ids: set
) -> Dict[str, str]:
    if not subtitles_paths:
        return {}

    if ijson is None:
        raise RuntimeError("Streaming requested but ijson is not installed. Install: pip install ijson")

    found: Dict[str, str] = {}
    processed = 0

    class NaNFixingReader:
        def __init__(self, file_obj):
            self.file_obj = file_obj
            self.pattern = re.compile(rb"\bNaN\b")
        def read(self, size=-1):
            chunk = self.file_obj.read(size)
            if not chunk:
                return chunk
            return self.pattern.sub(b"null", chunk)

    for p in subtitles_paths:
        with open(p, "rb") as raw_f:
            cleaned_f = NaNFixingReader(raw_f)

            for key, value in ijson.kvitems(cleaned_f, ""):
                vid = str(key)
                if vid not in wanted_ids:
                    continue


                txt = caption_segments_to_text(value)
                if txt:
                    found[vid] = txt
                    processed += 1
    logger.info(f"Found {len(found)} subtitles in total from {len(wanted_ids)} using list of subtitles files.")
    return found

def join_text_field(text_field: Any) -> str:

    if text_field is None:
        return ""

    if isinstance(text_field, str):
        return normalize_whitespace(text_field)

    if isinstance(text_field, list):
        parts: List[str] = []
        for item in text_field:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    # fallback: add any string values
                    for v in item.values():
                        if isinstance(v, str):
                            parts.append(v)
            else:
                parts.append(str(item))
        return normalize_whitespace("\n".join(parts))

    return normalize_whitespace(str(text_field))


def safe_filename(name: str) -> str:

    name = name.strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name if name else "item"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_yt_video_ids(yt_items: Any) -> List[str]:

    if not isinstance(yt_items, list):
        raise ValueError("yt_items must be a JSON list of objects like {'video_id': '...'}")

    out: List[str] = []
    seen = set()
    for idx, obj in enumerate(yt_items):
        if not isinstance(obj, dict) or "video_id" not in obj:
            logger.warning("Skipping invalid file1 item at index %d: %r", idx, obj)
            continue
        vid = obj["video_id"]
        if not isinstance(vid, str) or not vid.strip():
            logger.warning("Skipping empty/invalid video_id at index %d: %r", idx, obj)
            continue
        vid = vid.strip()
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


def iter_openi_records(openi_items: Any) -> List[Tuple[str, str, str]]:

    if not isinstance(openi_items, list):
        raise ValueError("OpenI file must be a JSON list of objects")

    out: List[Tuple[str, str, str]] = []
    seen = set()
    for idx, obj in enumerate(openi_items):
        if not isinstance(obj, dict):
            logger.warning("Skipping invalid OpenI item at index %d: %r", idx, obj)
            continue

        vid = obj.get("videoFileName")
        title = obj.get("title", "")
        if "videoTranscriptURL" not in obj:
            continue
        url = obj.get("videoTranscriptURL")

        if not isinstance(vid, str) or not vid.strip():
            logger.warning("Skipping OpenI item with invalid videoFileName at %d: %r", idx, obj)
            continue
        if not isinstance(url, str) or not url.strip():
            logger.warning("Skipping OpenI item with invalid videoTranscriptURL at %d: %r", idx, obj)
            continue

        vid = vid.strip()
        if vid in seen:
            continue
        seen.add(vid)

        if title is None:
            title = ""
        if not isinstance(title, str):
            title = str(title)

        out.append((vid, title.strip(), url.strip()))

    return out




def build_subtitles_map_for_ids_streaming(subtitles_path: str, wanted_ids: set) -> Dict[str, Any]:

    if ijson is None:
        raise RuntimeError("Streaming requested but ijson is not installed. Install: pip install ijson")

    found: Dict[str, Any] = {}

    class NaNFixingReader:
        def __init__(self, file_obj):
            self.file_obj = file_obj
            self.pattern = re.compile(rb'\bNaN\b')

        def read(self, size=-1):
            chunk = self.file_obj.read(size)
            if not chunk:
                return chunk
            return self.pattern.sub(b'null', chunk)

    with open(subtitles_path, "rb") as raw_f:
        cleaned_f = NaNFixingReader(raw_f)

        for key, value in ijson.kvitems(cleaned_f, ""):
            k = str(key)
            if k in wanted_ids:
                found[k] = value
                if len(found) % 2000 == 0:
                    logger.info(
                        "Collected %d/%d subtitle entries...",
                        len(found),
                        len(wanted_ids),
                    )

    logger.info("Subtitle entries found: %d. Missing: %d.", len(found), len(wanted_ids) - len(found))
    return found




@dataclass(frozen=True)
class HTTPConfig:
    timeout_sec: float = 20.0
    retries: int = 3
    backoff_sec: float = 1.0
    user_agent: str = "Mozilla/5.0 (compatible; build_corpus_cached/1.0)"


def make_session(cfg: HTTPConfig) -> requests.Session:

    s = requests.Session()
    s.headers.update({"User-Agent": cfg.user_agent})
    return s


def cache_paths(cache_dir: str, item_id: str) -> Tuple[str, str]:
    """
    Returns (text_path, meta_path) for cached transcript.
    """
    base = safe_filename(item_id)
    return (
        os.path.join(cache_dir, f"{base}.txt"),
        os.path.join(cache_dir, f"{base}.meta.json"),
    )


def load_cached_transcript(cache_dir: str, item_id: str) -> Optional[str]:
    txt_path, _ = cache_paths(cache_dir, item_id)
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw = f.read()
            return strip_timestamps(raw)
        except Exception as e:
            logger.warning("Failed reading cache %s: %s", txt_path, repr(e))
    return None


def save_cached_transcript(cache_dir: str, item_id: str, url: str, text: str) -> None:
    txt_path, meta_path = cache_paths(cache_dir, item_id)
    try:
        text = strip_timestamps(text)   # <-- add this line
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        meta = {"id": item_id, "url": url, "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed saving cache for %s: %s", item_id, repr(e))


def download_text_with_retries(session: requests.Session, url: str, cfg: HTTPConfig) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, cfg.retries + 1):
        try:
            resp = session.get(url, timeout=cfg.timeout_sec)
            resp.raise_for_status()
            resp.encoding = resp.encoding or "utf-8"
            return normalize_whitespace(resp.text)
        except Exception as e:
            last_err = e
            logger.debug("Download failed attempt %d/%d url=%s err=%s", attempt, cfg.retries, url, repr(e))
            if attempt < cfg.retries:
                time.sleep(cfg.backoff_sec * attempt)
    raise RuntimeError(f"Failed to download after {cfg.retries} attempts: {url}. Last error: {last_err!r}")


def get_transcript_cached(cache_dir: str, item_id: str, url: str, cfg: HTTPConfig) -> Tuple[str, bool]:

    cached = load_cached_transcript(cache_dir, item_id)
    if cached is not None:
        return normalize_whitespace(cached), True

    session = make_session(cfg)
    text = download_text_with_retries(session, url, cfg)
    save_cached_transcript(cache_dir, item_id, url, text)
    return text, False




def write_jsonl(out_path: str, records: Iterable[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json_array(out_path: str, records: Iterable[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for rec in records:
            if not first:
                f.write(",\n")
            first = False
            f.write(json.dumps(rec, ensure_ascii=False))
        f.write("\n]\n")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_youtube_video_corpus",  default="../data/VideoCorpus/YouTubeVideoCorpus.json", help="Path to JSON list with video_id items.")
    parser.add_argument("--path_to_howto100m_captions", default="../data/HowTo100M/raw_caption.json", help="Path to huge subtitles JSON dict keyed by video_id.")
    parser.add_argument("--path_to_openi_video_corpus", default="../data/VideoCorpus/OpenIVideoCorpus.json", help="Path to JSON list with videoFileName/title/videoTranscriptURL.")
    parser.add_argument("--path_to_save_processed_video_corpus", default="../data/VideoCorpus/Subtitles/VideoCorpus.jsonl", help="Output path (.jsonl recommended).")


    parser.add_argument("--cache-dir", default="../data/VideoCorpus/cache_transcripts", help="Directory to store transcript cache.")

    parser.add_argument("--workers", type=int, default=16, help="Number of download threads for transcripts.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries.")
    parser.add_argument("--backoff", type=float, default=1.0, help="HTTP backoff base seconds.")


    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()
    log_name = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(verbose=args.verbose, log_file=log_name)
    ensure_dir(args.cache_dir)
    http_cfg = HTTPConfig(timeout_sec=args.timeout, retries=args.retries, backoff_sec=args.backoff)

    yt_items = read_json(args.path_to_youtube_video_corpus)
    wanted_ids = iter_yt_video_ids(yt_items)
    wanted_set = set(wanted_ids)
    logger.info("YT unique video_ids: %d", len(wanted_ids))


    subtitles_map = build_subtitles_map_for_ids_streaming(args.path_to_howto100m_captions, wanted_set)

    openi_items = read_json(args.path_to_openi_video_corpus)
    openi_records = iter_openi_records(openi_items)
    logger.info("OpenI unique transcript items: %d", len(openi_records))

    def gen_yt_records() -> Iterator[Dict[str, str]]:
        missing = 0
        for vid in wanted_ids:
            payload = subtitles_map.get(vid)
            if not payload:
                missing += 1
                continue

            text_field = payload.get("text") if isinstance(payload, dict) else None
            contents = join_text_field(text_field)
            if contents:
                yield {"id": vid, "contents": contents}
        if missing:
            logger.info("YT missing/empty subtitles for %d ids.", missing)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(rec: Tuple[str, str, str]) -> Tuple[str, Optional[Dict[str, str]], Optional[str], bool]:
        video_file, title, url = rec
        try:
            transcript, from_cache = get_transcript_cached(args.cache_dir, video_file, url, http_cfg)
            combined = "\n".join([x for x in [title.strip(), transcript.strip()] if x])
            combined = normalize_whitespace(combined)
            if not combined:
                return video_file, None, "Empty combined contents", from_cache
            return video_file, {"id": video_file, "contents": combined}, None, from_cache
        except Exception as e:
            return video_file, None, repr(e), False

    def gen_OpenI_records_parallel() -> Iterator[Dict[str, str]]:
        if not openi_records:
            return
            yield  # for type checkers

        results: Dict[str, Tuple[Optional[Dict[str, str]], Optional[str], bool]] = {}
        cache_hits = 0
        failures = 0

        total = len(openi_records)
        bar = None
        if tqdm is not None:
            bar = tqdm(total=total, desc="Transcripts", unit="file")

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            future_to_id = {ex.submit(worker, rec): rec[0] for rec in openi_records}
            for fut in as_completed(future_to_id):
                vid = future_to_id[fut]
                rec_dict: Optional[Dict[str, str]] = None
                err: Optional[str] = None
                from_cache = False
                try:
                    _, rec_dict, err, from_cache = fut.result()
                except Exception as e:
                    err = repr(e)

                if from_cache:
                    cache_hits += 1
                if err is not None or rec_dict is None:
                    failures += 1
                    logger.warning("Transcript failed for %s: %s", vid, err)
                    results[vid] = (None, err, from_cache)
                else:
                    results[vid] = (rec_dict, None, from_cache)

                if bar is not None:
                    bar.update(1)

        if bar is not None:
            bar.close()

        logger.info("OpenI transcripts done. Cache hits: %d/%d. Failures: %d/%d.",
                    cache_hits, total, failures, total)

        for video_file, _, _ in openi_records:
            rec_dict, err, _ = results.get(video_file, (None, "Missing result", False))
            if rec_dict is not None:
                yield rec_dict

    def all_records() -> Iterator[Dict[str, str]]:
        yield from gen_yt_records()
        yield from gen_OpenI_records_parallel()


    write_jsonl(args.path_to_save_processed_video_corpus, all_records())

    logger.info("Done. Output written to: %s", args.path_to_save_processed_video_corpus)
    logger.info("Transcript cache dir: %s", os.path.abspath(args.cache_dir))
    return 0


if __name__ == "__main__":
    main()



