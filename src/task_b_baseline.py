import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
from util import save_json, load_json
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
        try:
            pmid = hit.docid
            content = json.loads(hit.lucene_document.get('raw'))['contents']
            docs_and_pmids.append((content, pmid))
        except (ValueError, AttributeError):
            continue
    return docs_and_pmids
def format_answer_prompt_for_llama(query, top_docs):
    prompt = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."

    prompt += f"\nQuestion: {query}"
    for i, (doc, pmid, _) in enumerate(top_docs):
        citation_n=i+1
        prompt += f"[{citation_n}]:({doc})\n"
    prompt += f"\n\nAnswer:"
    return prompt

def format_summary_prompt_for_llama(query, top_docs):
    prompt="Instruction: Write an accurate and concise summary of the provided document in relation to the given question. Focus on the information that directly helps answer the query and ignore irrelevant details. Use an unbiased and journalistic tone. Base the summary strictly on the document and do not add external knowledge."
    prompt_list=[]
    for i, (doc, pmid, _) in enumerate(top_docs):
        nprompt = f"{prompt}\nQuestion: {query}"
        nprompt += f"\nDocument: {doc}"
        nprompt += f"\n\nSummary:"
        prompt_list.append(nprompt)
    return prompt_list
def summarize_docs(query, retrieved_docs, llama_model, llama_tokenizer):
    summary_list=[]
    prompt_list = format_summary_prompt_for_llama(query, retrieved_docs)
    for prompt in prompt_list:
        inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = llama_model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        summary = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).split('Summary:')[1]
        summary_list.append(summary)
    summarized_docs=[]
    assert len(summary_list)==len(retrieved_docs)
    for i in range(len(summary_list)):
        summarized_docs.append((summary_list[i], retrieved_docs[i][1], retrieved_docs[i][2]))

    return summarized_docs
def generate_answer(query, retrieved_docs, llama_model, llama_tokenizer):

    prompt = format_answer_prompt_for_llama(query, retrieved_docs)
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    answer = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).split('Answer:')[1]

    return answer
def parse_response_with_citations(answer_text, citation_maps_pmid, citation_maps_video_id):
    def _expand_citation_numbers(citation_text: str) -> List[int]:
        numbers: List[int] = []
        for part in re.split(r"\s*,\s*", citation_text.strip()):
            if not part:
                continue
            if "-" in part:
                bounds = re.split(r"\s*-\s*", part)
                if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                    start, end = int(bounds[0]), int(bounds[1])
                    if start <= end:
                        numbers.extend(range(start, end + 1))
                    else:
                        numbers.extend(range(end, start + 1))
            elif part.isdigit():
                numbers.append(int(part))
        return numbers

    def _normalize_citation_map(
            citation_map: Optional[Union[Dict[Any, Any], List[Any]]]
    ) -> Dict[int, Any]:

        if citation_map is None:
            return {}
        if isinstance(citation_map, list):
            return {i + 1: v for i, v in enumerate(citation_map)}
        if isinstance(citation_map, dict):
            normalized: Dict[int, Any] = {}
            for k, v in citation_map.items():
                try:
                    normalized[int(k)] = v
                except (ValueError, TypeError):
                    continue
            return normalized
    if not isinstance(answer_text, str) or not answer_text.strip():
        return []

    pmid_map = _normalize_citation_map(citation_maps_pmid)
    video_map = _normalize_citation_map(citation_maps_video_id)
    answer_text = re.sub(r"[–—]", "-", answer_text)
    citation_pattern = re.compile(r"\[(\s*\d+(?:\s*[-,]\s*\d+)*\s*)\]")
    citation_matches = list(citation_pattern.finditer(answer_text))

    sentence_pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)", flags=re.DOTALL)
    sentence_spans = list(sentence_pattern.finditer(answer_text))
    responses: List[Dict[str, Any]] = []
    for span in sentence_spans:
        sentence = span.group().strip()
        if not sentence:
            continue
        sentence_start, sentence_end = span.start(), span.end()

        pmids: List[Any] = []
        video_ids: List[Any] = []
        seen_pmids = set()
        seen_video_ids = set()

        for match in citation_matches:
            if sentence_start <= match.start() < sentence_end:
                citation_text = match.group(1)
                citation_numbers = _expand_citation_numbers(citation_text)

                for idx in citation_numbers:
                    if idx in pmid_map and pmid_map[idx] not in seen_pmids:
                        pmids.append(pmid_map[idx])
                        seen_pmids.add(pmid_map[idx])

                    if idx in video_map and video_map[idx] not in seen_video_ids:
                        video_ids.append(video_map[idx])
                        seen_video_ids.add(video_map[idx])

        cleaned_sentence = citation_pattern.sub("", sentence)
        cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence).strip()

        cleaned_sentence = re.sub(r"\s+([.,;:!?])", r"\1", cleaned_sentence)

        if cleaned_sentence:
            responses.append(
                {
                    "text": cleaned_sentence,
                    "citations":{
                        "pmids": pmids,
                        "video_ids": video_ids,
                    }
                }
            )

    return responses
def get_pubmed_retrieved_only_baseline_results(path_to_topics, path_to_index, llama_model, llama_tokenizer, path_to_results):
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
        answer = generate_answer(query, top_10_docs, llama_model, llama_tokenizer)

        citation_maps_pmid = {}
        citation_maps_video_id = {}

        for i, (doc, pmid, _) in enumerate(top_10_docs):
            citation_maps_pmid[i + 1] = pmid
        responses = parse_response_with_citations(answer, citation_maps_pmid, citation_maps_video_id)


        results.append({'topic_id': topic_id, 'responses': responses})

    save_json(results, path_to_results)
    logger.info(f"Results saved to {path_to_results}")

def get_video_retrieved_only_baseline_results(path_to_topics, path_to_index, llama_model, llama_tokenizer, path_to_results):
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
        summ_top_10_videos = summarize_docs(query, top_10_videos, llama_model, llama_tokenizer)

        answer = generate_answer(query, summ_top_10_videos, llama_model, llama_tokenizer)

        citation_maps_pmid = {}
        citation_maps_video_id = {}

        for i, (doc, video_id, _) in enumerate(top_10_videos):
            citation_maps_video_id[i + 1] = video_id
        responses = parse_response_with_citations(answer, citation_maps_pmid, citation_maps_video_id)

        results.append({'topic_id': topic_id, 'responses': responses})

    save_json(results, path_to_results)
    logger.info(f"Results saved to {path_to_results}")

def get_pubmed_and_video_retrieved_baseline_results(path_to_topics, path_to_pubmed_index, path_to_video_subtitles_index, llama_model, llama_tokenizer, path_to_results):
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

        top_5_docs = reranked_docs[:5]
        top_5_videos = reranked_videos[:5]

        summ_top_5_videos = summarize_docs(query, top_5_videos, llama_model, llama_tokenizer)
        answer = generate_answer(query, top_5_docs+summ_top_5_videos, llama_model, llama_tokenizer)

        citation_maps_pmid = {}
        citation_maps_video_id = {}
        for i, (doc, pmid, _) in enumerate(top_5_docs):
            citation_maps_pmid[i + 1] = pmid
        for i, (doc, video_id, _) in enumerate(top_5_videos):
            citation_maps_video_id[i + len(citation_maps_pmid) + 1] = video_id
        responses = parse_response_with_citations(answer, citation_maps_pmid, citation_maps_video_id)

        results.append({'topic_id': topic_id, 'responses': responses})
    save_json(results, path_to_results)
    logger.info(f"Results saved to {path_to_results}")

def load_fine_tuned_model(adapter_path):
    llama_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    llama_model.to(device)
    logger.info(f"Model loaded from {adapter_path}")
    return llama_model, llama_tokenizer

def main() :
    parser = argparse.ArgumentParser(description="task-b baseline")
    parser.add_argument("--path_to_topics",  default="../data/TestDatasets/task_b_test.json")
    parser.add_argument("--path_to_pubmed_index", default="../data/indexes/pubmed_baseline_index")
    parser.add_argument("--path_to_videos_subtitles_index", default="../data/indexes/video_baseline_index")
    parser.add_argument("--path_to_fine_tuned_adapter",  default="../data/FineTunedAdapter")
    parser.add_argument("--path_to_save_results", default="../data/BaselineResults")



    args = parser.parse_args()
    log_name = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file=log_name)
    ensure_dir(args.path_to_save_results)

    llama_model, llama_tokenizer= load_fine_tuned_model(args.path_to_fine_tuned_adapter)
    get_pubmed_retrieved_only_baseline_results(args.path_to_topics, args.path_to_pubmed_index, llama_model, llama_tokenizer,
                                     path_to_results=os.path.join(args.path_to_save_results, "task_b_pubmed_retrieved_only_baseline_results.json"))
    get_video_retrieved_only_baseline_results(args.path_to_topics, args.path_to_videos_subtitles_index, llama_model, llama_tokenizer,
                                     path_to_results=os.path.join(args.path_to_save_results,
                                                                  "task_b_video_retrieved_only_baseline_results.json"))

    get_pubmed_and_video_retrieved_baseline_results(args.path_to_topics, args.path_to_pubmed_index, args.path_to_videos_subtitles_index, llama_model, llama_tokenizer,
                                                    path_to_results=os.path.join(args.path_to_save_results, "task_b_pubmed_and_video_retrieved_baseline_results.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())