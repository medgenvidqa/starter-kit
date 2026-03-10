import json
import sys
import logging
from pathlib import Path


EXPECTED_TOPICS = {f"B{i}" for i in range(1, 61)}


logger = logging.getLogger("submission_validator")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ValidationError(Exception):
    pass


def validate_list_of_strings(lst, name, topic_id):
    if not isinstance(lst, list):
        raise ValidationError(f"{topic_id}: {name} must be a list")
    for idx, item in enumerate(lst):
        if not isinstance(item, str):
            raise ValidationError(f"{topic_id}: {name}[{idx}] must be a string")

def validate_responses(responses, topic_id):
    if not isinstance(responses, list):
        raise ValidationError(f"{topic_id}: responses must be a list")

    for idx, response in enumerate(responses):
        if not isinstance(response, dict):
            raise ValidationError(f"{topic_id}: response {idx} must be an object")

        if "text" not in response:
            raise ValidationError(f"{topic_id}: response {idx} missing 'text'")
        if not isinstance(response["text"], str):
            raise ValidationError(f"{topic_id}: response {idx} 'text' must be string")

        if "citations" not in response:
            raise ValidationError(f"{topic_id}: response {idx} missing 'citations'")
        citations = response["citations"]

        if not isinstance(citations, dict):
            raise ValidationError(f"{topic_id}: response {idx} 'citations' must be object")

        if "pmids" not in citations:
            raise ValidationError(f"{topic_id}: response {idx} 'citations' missing 'pmids'")
        if "video_ids" not in citations:
            raise ValidationError(f"{topic_id}: response {idx} 'citations' missing 'video_ids'")

        validate_list_of_strings(citations["pmids"], "pmids", topic_id)
        validate_list_of_strings(citations["video_ids"], "video_ids", topic_id)


def validate_submission(file_path):
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

    logger.info("Validating topics...")
    seen_topics = set()

    for entry in data:
        if not isinstance(entry, dict):
            raise ValidationError("Each topic entry must be an object")

        if "topic_id" not in entry:
            raise ValidationError("Missing 'topic_id'")

        topic_id = entry["topic_id"]

        if topic_id not in EXPECTED_TOPICS:
            raise ValidationError(f"Invalid topic_id: {topic_id}")

        if topic_id in seen_topics:
            raise ValidationError(f"Duplicate topic_id detected: {topic_id}")
        seen_topics.add(topic_id)

        if "responses" not in entry:
            raise ValidationError(f"{topic_id}: missing 'responses'")

        validate_responses(entry["responses"], topic_id)

    missing_topics = EXPECTED_TOPICS - seen_topics
    if missing_topics:
        raise ValidationError(f"Missing topics: {sorted(missing_topics)}")

    logger.info("Submission validation completed successfully.")
    logger.info(f"Validated {len(seen_topics)} topics.")


def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python task_b_validation.py submission.json")
        sys.exit(1)
    try:
        validate_submission(sys.argv[1])
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()