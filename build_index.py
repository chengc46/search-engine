import os
import json
import re
from collections import defaultdict, Counter

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

CORPUS_ROOT = "./analyst"

def iter_json_files(root_dir: str):
    """
    Yield full paths to all .json files under root_dir.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".json"):
                yield os.path.join(dirpath, name)


def load_page(json_path: str) -> dict:
    """
    Load a single JSON file and return the parsed dict.
    Expected keys: 'url', 'content', 'encoding'
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_text_fields(html: str) -> dict:
    """
    Extract different text sections from HTML.
    For Milestone 1 you can just concatenate them,
    but we keep them separate in case you want importance later.
    Returns a dict with keys: title, headings, bold, body
    """
    soup = BeautifulSoup(html, "html.parser")

    #title
    title_tag = soup.title.string if soup.title and soup.title.string else ""
    title_text = title_tag or ""

    #headings h1–h3
    heading_texts = []
    for level in ["h1", "h2", "h3"]:
        for tag in soup.find_all(level):
            if tag.get_text():
                heading_texts.append(tag.get_text(separator=" ", strip=True))
    headings_text = " ".join(heading_texts)

    #bold / strong
    bold_texts = []
    for tag in soup.find_all(["b", "strong"]):
        if tag.get_text():
            bold_texts.append(tag.get_text(separator=" ", strip=True))
    bold_text = " ".join(bold_texts)

    #body text (everything)
    body_text = soup.get_text(separator=" ", strip=True)

    return {
        "title": title_text,
        "headings": headings_text,
        "bold": bold_text,
        "body": body_text,
    }

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    """
    Tokenize text into alphanumeric tokens, lowercased.
    """
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]

def stem_tokens(tokens, stemmer: PorterStemmer):
    """
    Apply Porter stemming to a list of tokens.
    """
    return [stemmer.stem(t) for t in tokens]

def build_index(corpus_root: str):
    """
    Walk through all JSON files in corpus_root and build:
      - inverted index: term -> {doc_id -> term_frequency}
      - doc_id_to_url: doc_id -> url
    Returns (index, doc_id_to_url)
    """
    stemmer = PorterStemmer()

    #inverted index: token -> {doc_id -> tf}
    index = defaultdict(dict)
    doc_id_to_url = {}

    doc_id = 0

    for json_path in iter_json_files(corpus_root):
        page = load_page(json_path)
        url = page.get("url", f"UNKNOWN_{doc_id}")
        html = page.get("content", "")

        #register document
        doc_id_to_url[doc_id] = url

        #extract text
        fields = extract_text_fields(html)

        #For M1: just concatenate everything
        full_text = " ".join(fields.values())

        #tokenize + stem
        tokens = tokenize(full_text)
        stemmed_tokens = stem_tokens(tokens, stemmer)

        #count term frequency in this document
        counter = Counter(stemmed_tokens)

        #update global index
        for term, tf in counter.items():
            index[term][doc_id] = tf

        doc_id += 1

    return index, doc_id_to_url


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def main():
    #Build index
    print("Building index...")
    index, doc_id_to_url = build_index(CORPUS_ROOT)
    print(f"Indexed {len(doc_id_to_url)} documents.")
    print(f"Unique tokens: {len(index)}")

    #Save to disk
    print("Saving index to disk...")
    save_json(index, "index.json")
    save_json(doc_id_to_url, "doc_ids.json")

    #Compute basic analytics
    index_size_bytes = os.path.getsize("index.json")
    index_size_kb = index_size_bytes / 1024

    print("=== Analytics for Milestone 1 ===")
    print(f"Number of documents: {len(doc_id_to_url)}")
    print(f"Number of unique tokens: {len(index)}")
    print(f"Index size on disk: {index_size_kb:.2f} KB")


if __name__ == "__main__":
    main()
