import json
import math
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set

from nltk.stem import PorterStemmer


# Paths
INDEX_PATH = "index.json"
DOCIDS_PATH = "doc_ids.json"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


# Loading
def load_index(path: str) -> Dict[str, Dict[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # convert doc_id keys back to int
    return {t: {int(d): tf for d, tf in postings.items()} for t, postings in raw.items()}


def load_doc_ids(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(d): url for d, url in raw.items()}


# Query Processing
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def stem_tokens(tokens: List[str], stemmer: PorterStemmer) -> List[str]:
    return [stemmer.stem(t) for t in tokens]


# TF-IDF Utilities
def compute_idf(index: Dict[str, Dict[int, int]], num_docs: int) -> Dict[str, float]:
    """
    idf(t) = log((N + 1) / (df_t + 1)) + 1
    """
    idf = {}
    for term, postings in index.items():
        df = len(postings)
        idf[term] = math.log((num_docs + 1) / (df + 1)) + 1.0
    return idf


def compute_doc_lengths(index: Dict[str, Dict[int, int]]) -> Dict[int, int]:
    """
    Total number of terms per document (for normalization).
    """
    lengths = defaultdict(int)
    for postings in index.values():
        for doc_id, tf in postings.items():
            lengths[doc_id] += tf
    return lengths


# Retrieval
def soft_and_retrieval(
    query_terms: List[str],
    index: Dict[str, Dict[int, int]],
) -> Set[int]:
    """
    Soft AND: documents must contain at least half of the query terms.
    Ensures recall for queries like 'ICS 33 syllabus'.
    """
    doc_matches = defaultdict(int)

    for term in query_terms:
        for doc_id in index.get(term, {}):
            doc_matches[doc_id] += 1

    min_required = max(1, len(query_terms) // 2)
    return {doc for doc, count in doc_matches.items() if count >= min_required}


def score_tf_idf(
    doc_ids: Set[int],
    query_terms: List[str],
    index: Dict[str, Dict[int, int]],
    idf: Dict[str, float],
    doc_lengths: Dict[int, int],
) -> List[Tuple[int, float]]:
    scores = defaultdict(float)

    for term in query_terms:
        postings = index.get(term, {})
        term_idf = idf.get(term, 0.0)
        for doc_id in doc_ids:
            tf = postings.get(doc_id, 0)
            scores[doc_id] += tf * term_idf

    # Normalize by document length
    for doc_id in scores:
        scores[doc_id] /= (1 + math.log(1 + doc_lengths.get(doc_id, 1)))

    return sorted(scores.items(), key=lambda x: -x[1])


def search(
    query: str,
    index,
    doc_id_to_url,
    stemmer,
    idf,
    doc_lengths,
    top_k=5,
):
    tokens = tokenize(query)
    if not tokens:
        return []

    stemmed = stem_tokens(tokens, stemmer)

    matching_docs = soft_and_retrieval(stemmed, index)
    if not matching_docs:
        return []

    ranked = score_tf_idf(
        matching_docs,
        stemmed,
        index,
        idf,
        doc_lengths
    )

    return [(doc_id_to_url[d], score) for d, score in ranked[:top_k]]


# CLI Interface
def main():
    print("Loading index and document mappings...")
    index = load_index(INDEX_PATH)
    doc_id_to_url = load_doc_ids(DOCIDS_PATH)
    stemmer = PorterStemmer()

    print(f"Loaded index with {len(index)} terms.")
    print(f"Loaded {len(doc_id_to_url)} documents.")

    print("Precomputing IDF and document lengths...")
    idf = compute_idf(index, len(doc_id_to_url))
    doc_lengths = compute_doc_lengths(index)

    print("\nICS Search Engine — Milestone 3")
    print("Type your query. Press Enter to quit.")
    print("-" * 60)

    while True:
        query = input("\nQuery> ").strip()
        if not query:
            print("Exiting.")
            break

        start = time.time()
        results = search(
            query,
            index,
            doc_id_to_url,
            stemmer,
            idf,
            doc_lengths
        )
        elapsed = (time.time() - start) * 1000

        if not results:
            print("No results found.")
            continue

        print(f"Top {len(results)} results (time={elapsed:.2f} ms):")
        for i, (url, score) in enumerate(results, start=1):
            print(f"{i}. {url}  (score={score:.2f})")


if __name__ == "__main__":
    main()
