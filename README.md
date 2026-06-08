# Search Engine & Web Crawler

A Python-based search engine that indexes webpages and retrieves documents using information retrieval techniques such as inverted indexing and TF-IDF ranking.

This project was developed as part of an Information Retrieval course at the University of California, Irvine.

## Example Result
<img width="1176" height="645" alt="image" src="https://github.com/user-attachments/assets/1f1203d2-e842-40d4-b0c9-6ed86081cce9" />

## Features

* Build an inverted index from a collection of webpages
* Process and normalize search queries
* Apply tokenization and stemming
* Rank results using TF-IDF scoring
* Retrieve relevant documents efficiently

## Technologies

* Python
* NLTK
* BeautifulSoup
* JSON
* Information Retrieval

## Project Structure

```text
search-engine/
├── build_index.py
├── search_engine.py
├── requirements.txt
└── data/
    ├── index.json
    └── doc_ids.json
```

### build_index.py

Parses webpage data and generates the inverted index used by the search engine.

### search_engine.py

Processes user queries and retrieves ranked search results.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the search engine:

```bash
python search_engine.py
```

## Notes

The original webpage corpus is not included in this repository due to course dataset distribution restrictions.

The provided index files are included for demonstration purposes.

## Author

Cheng Chen

B.S. Informatics, Human-Computer Interaction

University of California, Irvine
