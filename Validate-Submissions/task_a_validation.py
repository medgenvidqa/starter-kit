import json
import sys
import logging
from pathlib import Path


EXPECTED_TOPICS = {f"A{i}" for i in range(1, 61)}


logger = logging.getLogger("submission_validator")
handler = logging.StreamHandler()

formatter = logging.Formatter(
    "[%(levelname)s] %(message)s"
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



class ValidationError(Exception):
    pass


def validate_score(score, location):
    if not isinstance(score, (int, float)):
        raise ValidationError(f"{location}: score must be numeric")
    if score < 0:
        logger.warning(f"{location}: score is negative")


def validate_documents(docs, topic_id):
    if not isinstance(docs, list):
        raise ValidationError(f"{topic_id}: relevant_documents must be a list")

    if len(docs) == 0:
        logger.warning(f"{topic_id}: no documents provided")
    seen_ids = set()

    for idx, item in enumerate(docs):
        if not isinstance(item, dict):
            raise ValidationError(
                f"{topic_id}: document entry {idx} must be an object"
            )

        if "doc_id" not in item:
            raise ValidationError(
                f"{topic_id}: document entry {idx} missing 'doc_id'"
            )

        if "relevant_score" not in item:
            raise ValidationError(
                f"{topic_id}: document entry {idx} missing 'relevant_score'"
            )

        doc_id = item["doc_id"]

        if not isinstance(doc_id, str):
            raise ValidationError(
                f"{topic_id}: doc_id must be a string"
            )

        if doc_id in seen_ids:
            logger.warning(
                f"{topic_id}: duplicate document id detected ({doc_id})"
            )

        seen_ids.add(doc_id)

        validate_score(item["relevant_score"], f"{topic_id} document {doc_id}")


def validate_videos(videos, topic_id):

    if not isinstance(videos, list):
        raise ValidationError(f"{topic_id}: relevant_videos must be a list")

    if len(videos) == 0:
        logger.warning(f"{topic_id}: no videos provided")

    seen_ids = set()

    for idx, item in enumerate(videos):

        if not isinstance(item, dict):
            raise ValidationError(
                f"{topic_id}: video entry {idx} must be an object"
            )

        if "video_id" not in item:
            raise ValidationError(
                f"{topic_id}: video entry {idx} missing 'video_id'"
            )

        if "relevant_score" not in item:
            raise ValidationError(
                f"{topic_id}: video entry {idx} missing 'relevant_score'"
            )

        video_id = item["video_id"]

        if not isinstance(video_id, str):
            raise ValidationError(
                f"{topic_id}: video_id must be a string"
            )

        if video_id in seen_ids:
            logger.warning(
                f"{topic_id}: duplicate video id detected ({video_id})"
            )

        seen_ids.add(video_id)

        validate_score(item["relevant_score"], f"{topic_id} video {video_id}")




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
        raise ValidationError("Submission root must be a JSON list")

    logger.info("Validating topic structure...")

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

        if "relevant_documents" not in entry:
            raise ValidationError(
                f"{topic_id}: missing 'relevant_documents'"
            )

        if "relevant_videos" not in entry:
            raise ValidationError(
                f"{topic_id}: missing 'relevant_videos'"
            )

        validate_documents(entry["relevant_documents"], topic_id)
        validate_videos(entry["relevant_videos"], topic_id)

    missing_topics = EXPECTED_TOPICS - seen_topics

    if missing_topics:
        raise ValidationError(
            f"Missing topics: {sorted(missing_topics)}"
        )

    logger.info("Submission validation completed successfully.")
    logger.info(f"Validated {len(seen_topics)} topics.")




def main():

    if len(sys.argv) != 2:
        logger.error(
            "Usage: python validate_submission.py submission.json"
        )
        sys.exit(1)

    try:
        validate_submission(sys.argv[1])
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()