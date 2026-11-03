"""Copy five articles from data/corpus.json into data/sample_articles.json, for writing custom questions.

The five paragraphs link to each other (film -> director, film -> novel author), so they
support two-hop questions like HotpotQA's. Open data/sample_articles.json to read them;
the "example_questions" list holds questions written from them, with gold answers, to type
into the UI's custom-question box. The agents still search the whole corpus (66,581
paragraphs), not just these five.

Usage:  python make_sample_articles.py      (needs data/corpus.json from select_questions.py)
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

TITLES = [
    "The Prestige (film)",
    "Christopher Nolan",
    "Rumble Fish",
    "Francis Ford Coppola",
    "The Outsiders (novel)",
]

EXAMPLE_QUESTIONS = [
    {
        "type": "bridge",
        "question": "In what year was the director of The Prestige born?",
        "answer": "1970",
        "uses": ["The Prestige (film)", "Christopher Nolan"],
        "reasoning": "The Prestige was 'directed by Christopher Nolan'; Nolan was 'born 30 July 1970'.",
    },
    {
        "type": "bridge",
        "question": "The film Rumble Fish is based on a novel whose author also wrote which coming-of-age "
                    "novel published in 1967?",
        "answer": "The Outsiders",
        "uses": ["Rumble Fish", "The Outsiders (novel)"],
        "reasoning": "Rumble Fish is 'based on the novel \"Rumble Fish\" by S. E. Hinton'; 'The Outsiders is a "
                     "coming-of-age novel by S. E. Hinton, first published in 1967'.",
    },
    {
        "type": "comparison",
        "question": "Who was born first, Christopher Nolan or Francis Ford Coppola?",
        "answer": "Francis Ford Coppola",
        "uses": ["Christopher Nolan", "Francis Ford Coppola"],
        "reasoning": "Nolan: born 30 July 1970; Coppola: born April 7, 1939.",
    },
]


def main() -> None:
    corpus = {d["title"]: d["text"] for d in json.loads((HERE / "data" / "corpus.json").read_text())}
    missing = [t for t in TITLES if t not in corpus]
    if missing:
        raise SystemExit(f"not in corpus: {missing}")
    out = {
        "note": "Five of the 66,581 paragraphs in data/corpus.json (first paragraph of each Wikipedia "
                "article, 2017). The agents search the whole corpus, not only these five.",
        "articles": [{"title": t, "text": corpus[t]} for t in TITLES],
        "example_questions": EXAMPLE_QUESTIONS,
    }
    path = HERE / "data" / "sample_articles.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
