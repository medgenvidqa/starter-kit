import json
import sys
import logging
from pathlib import Path
import re


EXPECTED_TOPICS = {f"C{i}" for i in range(1, 6)}


logger = logging.getLogger("submission_validator")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ValidationError(Exception):
    pass


TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")

def validate_time_format(time_str, field_name, topic_id):
    if not isinstance(time_str, str):
        raise ValidationError(f"{topic_id}: {field_name} must be a string")
    if not TIME_PATTERN.match(time_str):
        raise ValidationError(f"{topic_id}: {field_name} '{time_str}' must be in MM:SS format")


def validate_taskC_submission(file_path):
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(f"Submission file not found: {file_path}")

    logger.info(f"Loading submission file: {file_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValidationError(f"Invalid JSON file: {e}")

    if not isinstance(data, list):
        raise ValidationError("Submission root must be a list")

    logger.info("Validating Task C topics...")
    seen_ids = set()

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValidationError(f"Entry {idx} must be an object")

        if "id" not in entry:
            raise ValidationError(f"Entry {idx} missing 'id'")
        topic_id = entry["id"]

        if topic_id not in EXPECTED_TOPICS:
            raise ValidationError(f"Invalid topic id: {topic_id}")

        if topic_id in seen_ids:
            raise ValidationError(f"Duplicate topic id detected: {topic_id}")
        seen_ids.add(topic_id)

        if "answer_start" not in entry:
            raise ValidationError(f"{topic_id}: missing 'answer_start'")
        if "answer_end" not in entry:
            raise ValidationError(f"{topic_id}: missing 'answer_end'")

        validate_time_format(entry["answer_start"], "answer_start", topic_id)
        validate_time_format(entry["answer_end"], "answer_end", topic_id)

    missing_ids = EXPECTED_TOPICS - seen_ids
    if missing_ids:
        raise ValidationError(f"Missing topics: {sorted(missing_ids)}")

    logger.info("Task C submission validation completed successfully.")
    logger.info(f"Validated {len(seen_ids)} topics.")


def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python task_c_validation.py submission.json")
        sys.exit(1)

    try:
        validate_taskC_submission(sys.argv[1])
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()