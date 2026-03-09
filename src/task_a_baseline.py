import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
from util import save_json, load_json
import re, json, random
import numpy as np
import argparse
import json
import logging
from torch.nn.functional import sigmoid

import os

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
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker_model.to(device)
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



def rerank_documents(query, docs_and_pmids):

    inputs = [(query, doc) for doc, _ in docs_and_pmids]
    encoded = reranker_tokenizer.batch_encode_plus(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = reranker_model(**encoded).logits.squeeze(-1)

    scores = sigmoid(outputs) if outputs.ndim == 1 else outputs

    scored_docs = []
    for (doc, pmid), score in zip(docs_and_pmids, scores.cpu().numpy()):
        scored_docs.append((doc, pmid, float(score)))

    scored_docs.sort(key=lambda x: x[2], reverse=True)
    return scored_docs




def retrieve_top_docs_and_pmids(query, lucene_bm25_searcher, top_k=100):

    hits = lucene_bm25_searcher.search(query, k=top_k)
    docs_and_pmids = []

    for hit in hits:
        pmid = hit.docid
        content = None

        try:
            raw = hit.lucene_document.get('raw')
            if raw:
                content = json.loads(raw).get('contents')
        except (ValueError, AttributeError, TypeError):
            content = None

        if not content:
            logger.info("Please include the contents in the index. If you are using video subtitles index, first prepare the raw file and build the index including raw file.")
            exit(-1)

        if content:
            docs_and_pmids.append((content, pmid))

    return docs_and_pmids



def get_pubmed_retrieved_only_baseline_results(path_to_topics, path_to_index, path_to_results):
    lucene_bm25_searcher = LuceneSearcher(path_to_index)
    data = load_json(path_to_topics)
    results = []

    for item in tqdm(data):
        topic_id = str(item['id'])
        query = item['question']
        logger.info(f"Processing Question ID {topic_id}...")
        docs_and_pmids = retrieve_top_docs_and_pmids(query, lucene_bm25_searcher)
        reranked_docs = rerank_documents(query, docs_and_pmids)
        top_10_docs = reranked_docs[:10]
        relevant_docs=[]
        for (doc, pmid, score) in top_10_docs:
            relevant_docs.append({'doc_id': pmid, 'relevant_score': score})
        results.append({'topic_id': topic_id, 'relevant_documents': relevant_docs,
                        'relevant_videos': None})

    save_json(results, path_to_results)
    logger.info(f"Results saved to {path_to_results}")

def get_video_retrieved_only_baseline_results(path_to_topics, path_to_index, path_to_results):
    lucene_bm25_searcher = LuceneSearcher(path_to_index)
    data = load_json(path_to_topics)
    results = []
    for item in tqdm(data):
        topic_id = str(item['id'])
        query = item['question']
        logger.info(f"Processing Question ID {topic_id}...")
        videos_and_vids = retrieve_top_docs_and_pmids(query, lucene_bm25_searcher)

        reranked_videos = rerank_documents(query, videos_and_vids)
        top_10_videos = reranked_videos[:10]
        relevant_videos=[]
        for (video, vid, score) in top_10_videos:
            relevant_videos.append({'video_id': vid, 'relevant_score': score})
        results.append({'topic_id': topic_id, 'relevant_documents': None,
                        'relevant_videos': relevant_videos})

    save_json(results, path_to_results)
    logger.info(f"Results saved to {path_to_results}")

def get_pubmed_and_video_retrieved_baseline_results(path_to_topics, path_to_pubmed_index, path_to_video_subtitles_index, path_to_results):
    pubmed_lucene_bm25_searcher = LuceneSearcher(path_to_pubmed_index)
    video_subtitles_lucene_bm25_searcher = LuceneSearcher(path_to_video_subtitles_index)

    data = load_json(path_to_topics)
    results = []
    for item in tqdm(data):
        topic_id = str(item['id'])
        query = item['question']
        logger.info(f"Processing Question ID {topic_id}...")
        docs_and_pmids = retrieve_top_docs_and_pmids(query, pubmed_lucene_bm25_searcher)
        videos_and_vids = retrieve_top_docs_and_pmids(query, video_subtitles_lucene_bm25_searcher)

        reranked_docs = rerank_documents(query, docs_and_pmids)
        reranked_videos = rerank_documents(query, videos_and_vids)

        top_10_docs = reranked_docs[:10]
        top_10_videos = reranked_videos[:10]
        relevant_docs = []
        for (doc, pmid, score) in top_10_docs:
            relevant_docs.append({'doc_id': pmid, 'relevant_score': score})
        relevant_videos=[]
        for (doc, vid, score) in top_10_videos:
            relevant_videos.append({'video_id': vid, 'relevant_score': score})
        results.append({'topic_id': topic_id, 'relevant_documents': relevant_docs,
                        'relevant_videos': relevant_videos})

    save_json(results, path_to_results)
    logger.info(f"Results saved to {path_to_results}")


def main() :
    parser = argparse.ArgumentParser(description="task-a baseline")
    parser.add_argument("--path_to_topics",  default="../data/TestDatasets/task_a_test.json")
    parser.add_argument("--path_to_pubmed_index", default="../data/indexes/pubmed_baseline_index")
    parser.add_argument("--path_to_videos_subtitles_index", default="../data/indexes/video_baseline_index")
    parser.add_argument("--path_to_save_results", default="../data/BaselineResults")



    args = parser.parse_args()
    log_name = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file=log_name)
    ensure_dir(args.path_to_save_results)
    get_pubmed_retrieved_only_baseline_results(args.path_to_topics, args.path_to_pubmed_index,
                                     path_to_results=os.path.join(args.path_to_save_results, "task_a_pubmed_retrieved_only_baseline_results.json"))
    get_video_retrieved_only_baseline_results(args.path_to_topics, args.path_to_videos_subtitles_index,
                                     path_to_results=os.path.join(args.path_to_save_results,
                                                                  "task_a_video_retrieved_only_baseline_results.json"))

    get_pubmed_and_video_retrieved_baseline_results(args.path_to_topics, args.path_to_pubmed_index, args.path_to_videos_subtitles_index,
                                                    path_to_results=os.path.join(args.path_to_save_results, "task_a_pubmed_and_video_retrieved_baseline_results.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())