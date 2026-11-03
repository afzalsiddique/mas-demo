"""Run the three demo conditions and write results/.

    python run_experiment.py                 # auto-match k to the team's token budget
    python run_experiment.py --k 5           # fix k by hand
    python run_experiment.py --repeats 1 --limit 2   # quick smoke test

Procedure (per repeat r = 0..R-1, per question):
  1. Team run (condition C) and one single-agent sample (condition A).
  2. k = round(mean team tokens / mean single-sample tokens), unless --k is given.
  3. k-1 more single-agent samples; condition B = majority vote over the k samples
     (sample #1 is the condition-A answer, so A and B share that sample).
All runs use the same model, temperature, tools and questions.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from pathlib import Path

import mas_demo as md

RESULTS = md.HERE / "results"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=None, help="vote size; default: matched to team tokens")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N questions")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = md.load_questions(experiment_only=True)[: args.limit]
    client = md.make_client()
    single, team = md.build_single(client), md.build_team(client)
    print(f"server={md.resolve_base_url()} model={md.MODEL_NAME} T={md.TEMPERATURE} "
          f"questions={len(questions)} repeats={args.repeats}", flush=True)

    runs: list[dict] = []

    async def job(kind: str, r: int, q: dict, sample: int = 0) -> dict:
        agent, turns = (team, 12) if kind == "team" else (single, 10)
        res = await md.run_agent(agent, q, max_turns=turns)
        rec = {"kind": kind, "repeat": r, "sample": sample, "qid": q["qid"], "type": q["type"],
               "question": q["question"], "gold": q["answer"], **res}
        print(f"  {kind:6s} r={r} s={sample} {q['qid'][:8]} calls={rec['calls']:2d} "
              f"tok={rec['total_tokens']:6d} pred={md._short(rec['prediction'], 40)!r} "
              f"em={md.exact_match(rec['prediction'], q['answer'])}"
              + (f" ERR={rec['error'][:60]}" if rec["error"] else ""), flush=True)
        return rec

    # ---- phase 1: team + first single sample -------------------------------------------
    t0 = time.time()
    print("phase 1: team runs + one single-agent sample each", flush=True)
    jobs = [job(kind, r, q) for r in range(args.repeats) for q in questions for kind in ("team", "single")]
    runs += await md.gather_limited(jobs, args.concurrency)

    team_tok = statistics.mean(x["total_tokens"] for x in runs if x["kind"] == "team")
    single_tok = statistics.mean(x["total_tokens"] for x in runs if x["kind"] == "single")
    k = args.k or max(args.k_min, min(args.k_max, round(team_tok / max(1, single_tok))))
    print(f"mean tokens: team={team_tok:.0f} single={single_tok:.0f}  ->  k={k}", flush=True)

    # ---- phase 2: remaining single samples ---------------------------------------------
    print(f"phase 2: {k - 1} more single-agent samples per question/repeat", flush=True)
    jobs = [job("single", r, q, s) for r in range(args.repeats) for q in questions for s in range(1, k)]
    runs += await md.gather_limited(jobs, args.concurrency)
    print(f"done in {time.time() - t0:.0f}s, {len(runs)} runs", flush=True)

    # ---- write raw runs ----------------------------------------------------------------
    with open(out / "runs.jsonl", "w") as f:
        for x in runs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # ---- per-condition rows ------------------------------------------------------------
    rows = []
    for r in range(args.repeats):
        for q in questions:
            mine = [x for x in runs if x["repeat"] == r and x["qid"] == q["qid"]]
            tm = next(x for x in mine if x["kind"] == "team")
            samples = sorted((x for x in mine if x["kind"] == "single"), key=lambda x: x["sample"])
            a = samples[0]
            vote = md.majority_vote([x["prediction"] for x in samples])
            for cond, pred, used in (("A_single", a["prediction"], [a]),
                                     (f"B_vote@{k}", vote, samples),
                                     ("C_team", tm["prediction"], [tm])):
                rows.append({
                    "repeat": r, "qid": q["qid"], "type": q["type"], "condition": cond,
                    "prediction": pred, "gold": q["answer"],
                    "em": md.exact_match(pred, q["answer"]), "f1": md.f1(pred, q["answer"]),
                    "calls": sum(x["calls"] for x in used),
                    "input_tokens": sum(x["input_tokens"] for x in used),
                    "output_tokens": sum(x["output_tokens"] for x in used),
                    "total_tokens": sum(x["total_tokens"] for x in used),
                    "errors": sum(1 for x in used if x["error"]),
                })
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    meta = {"k": k, "repeats": args.repeats, "n_questions": len(questions), "model": md.MODEL_NAME,
            "temperature": md.TEMPERATURE, "max_tokens_per_call": md.MAX_TOKENS_PER_CALL,
            "mean_tokens_team_run": round(team_tok), "mean_tokens_single_sample": round(single_tok)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out/'runs.jsonl'}, {out/'results.csv'}, {out/'meta.json'}")


if __name__ == "__main__":
    asyncio.run(main())
