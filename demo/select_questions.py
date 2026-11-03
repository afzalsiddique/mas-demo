"""Pick the demo questions from the HotpotQA dev set (distractor setting).

The selection is rule-based and seeded, NOT hand-picked by looking at model
results, so the comparison is not cherry-picked:
  * level == "hard" (all dev questions are "hard")
  * short gold answer (<= 4 words) so exact-match scoring and voting are clean
  * the gold answer string appears in the provided paragraphs (answerable)
  * "bridge" questions (two hops: find entity X, then a fact about X)
    and "comparison" questions (look up two entities, compare them)
  * random.Random(SEED) sampling

Two groups of questions:
  * the EXPERIMENT set (3 bridge + 2 comparison, seed 7): the questions behind the
    pre-computed results in results/ (run_experiment.py uses only these)
  * EXTRA questions for the live UI (default 9 bridge + 6 comparison), drawn with a
    second seed from the questions not already picked, so adding extras never
    changes the experiment set
  * DEMO_PICKS (listed first in the file and the UI): a few questions that WERE hand-picked by looking at model results
    (screened over 65+30 candidates), because each shows one effect clearly in the
    live demo (team wins / team loses). They illustrate; they are not evidence.
    The unbiased numbers are the experiment set's (results/). See docs/demo-script.md

Outputs:
  data/questions.json   one record per question with its 10 paragraphs
                        (2 gold + 8 distractors); for these questions the agents
                        can search only those 10 paragraphs
  data/corpus.json      every distinct paragraph in the dev set (~66,600 Wikipedia
                        intro paragraphs); the search collection for custom
                        questions typed into the UI (not in git: regenerate it)

Usage:  python select_questions.py [--n-bridge 3] [--n-comparison 2] [--seed 7]
                                   [--extra-bridge 9] [--extra-comparison 6]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent

# dev-set indices, hand-picked for the live demo; listed FIRST (questions 1-3 in the UI); see the docstring
DEMO_PICKS = [
    (783, "comparison"),   # Ulrich Walter or Luca Parmitano, who is younger? single 0/16, team 9/12
    (2010, "bridge"),      # ... sixth studio album in 2012 -> Psy. single 16/16, team 0/12 ("Lecrae")
    (2393, "bridge"),      # Julien Kang's brother's nationality -> Canadian. backup for 2010
]


def answerable(row) -> bool:
    gold = row["answer"].lower()
    text = " ".join("".join(s) for s in row["context"]["sentences"]).lower()
    return gold in text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bridge", type=int, default=3)
    ap.add_argument("--n-comparison", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--extra-bridge", type=int, default=9)
    ap.add_argument("--extra-comparison", type=int, default=6)
    ap.add_argument("--extra-seed", type=int, default=2026)
    ap.add_argument("--out", default=str(HERE / "data" / "questions.json"))
    ap.add_argument("--corpus-out", default=str(HERE / "data" / "corpus.json"))
    args = ap.parse_args()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    pools = {
        qtype: [
            i
            for i, r in enumerate(ds)
            if r["type"] == qtype
            and r["level"] == "hard"
            and len(r["answer"].split()) <= 4
            and r["answer"].lower() not in {"yes", "no"}  # keep answers non-trivial to vote on
            and answerable(r)
        ]
        for qtype in ("bridge", "comparison")
    }

    # experiment set: exactly the original procedure (same rng call sequence)
    rng = random.Random(args.seed)
    picked = []
    for qtype, n in (("bridge", args.n_bridge), ("comparison", args.n_comparison)):
        picked += [(qtype, i, True) for i in rng.sample(pools[qtype], n)]

    # extras: a second rng, never touching the experiment questions
    rng2 = random.Random(args.extra_seed)
    taken = {i for _, i, _ in picked}
    for qtype, n in (("bridge", args.extra_bridge), ("comparison", args.extra_comparison)):
        picked += [(qtype, i, False) for i in rng2.sample([i for i in pools[qtype] if i not in taken], n)]
    n_sampled = len(picked)
    picked += [(qtype, i, False) for i, qtype in DEMO_PICKS]

    records = []
    for n, (qtype, i, in_exp) in enumerate(picked):
        r = ds[i]
        assert r["type"] == qtype
        docs = [
            {"title": t, "text": "".join(s).strip()}
            for t, s in zip(r["context"]["title"], r["context"]["sentences"])
        ]
        records.append(
            {
                "qid": r["id"],
                "dev_index": i,
                "type": qtype,
                "experiment": in_exp,
                "demo_pick": n >= n_sampled,
                "question": r["question"],
                "answer": r["answer"],
                "gold_titles": sorted(set(r["supporting_facts"]["title"])),
                "docs": docs,
            }
        )

    # UI order: demo picks first (1-3), then the experiment set (4-8), then the extras (9-23).
    # Sampling above is unchanged, so the same questions are picked; only the listing order moves.
    records = [r for r in records if r["demo_pick"]] + [r for r in records if not r["demo_pick"]]
    Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False))
    for rec in records:
        print(f"[{'exp' if rec['experiment'] else 'demo' if rec['demo_pick'] else 'extra':5s} {rec['type']:10s}] {rec['question']}  =>  {rec['answer']}")
    print(f"wrote {len(records)} questions to {args.out}")

    corpus: dict[str, str] = {}
    for ctx in ds["context"]:
        for t, s in zip(ctx["title"], ctx["sentences"]):
            corpus.setdefault(t, "".join(s).strip())
    Path(args.corpus_out).write_text(
        json.dumps([{"title": t, "text": x} for t, x in corpus.items()], ensure_ascii=False))
    print(f"wrote {len(corpus)} paragraphs to {args.corpus_out}")


if __name__ == "__main__":
    main()
