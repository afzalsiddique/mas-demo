"""Summarize results/ into a table, a plot and a readable trace for the slides.

    python analyze.py            # reads results/results.csv, runs.jsonl, meta.json

Writes:
  results/summary.md               condition table (accuracy, calls, tokens)
  results/accuracy_vs_tokens.png   accuracy vs. tokens per question
  results/trace_example.md         one team trace and one single-agent trace
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# Validated reference palette (first three categorical slots, light mode) + surface/ink tokens.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # A single, B vote@k, C team (fixed order)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    order = sorted(df["condition"].unique(), key=lambda c: c[0])  # A_, B_, C_
    rows = []
    for c in order:
        d = df[df.condition == c]
        per_rep = d.groupby("repeat")["em"].mean()  # accuracy of each repeat
        rows.append({
            "condition": c,
            "accuracy_em": d.em.mean(),
            "acc_sd_over_repeats": per_rep.std(ddof=1) if len(per_rep) > 1 else 0.0,
            "f1": d.f1.mean(),
            "calls_per_q": d.calls.mean(),
            "tokens_per_q": d.total_tokens.mean(),
            "tok_sd_over_repeats": d.groupby("repeat").total_tokens.mean().std(ddof=1) if len(per_rep) > 1 else 0.0,
            "errors": int(d.errors.sum()),
        })
    return pd.DataFrame(rows)


def plot(s: pd.DataFrame, meta: dict, path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    labels = {"A": "Single agent (1 try)", "B": f"Single agent, vote of {meta['k']}", "C": "Team (manager + 3)"}
    for i, r in s.reset_index(drop=True).iterrows():
        key = r.condition[0]
        x, y = r.tokens_per_q / 1000, r.accuracy_em * 100
        ax.errorbar(x, y, xerr=r.tok_sd_over_repeats / 1000, yerr=r.acc_sd_over_repeats * 100,
                    fmt="none", ecolor=SERIES[i], elinewidth=1.5, capsize=0, alpha=0.6, zorder=2)
        ax.scatter([x], [y], s=150, color=SERIES[i], edgecolor=SURFACE, linewidth=2, zorder=3, label=labels[key])
        # A and B: label to the right; C: label below (B and C usually sit close together)
        off, ha, va = {"C": ((0, -34), "center", "top")}.get(key, ((12, 0), "left", "center"))
        ax.annotate(f"{labels[key]}\n{y:.0f}% · {r.calls_per_q:.1f} calls · {r.tokens_per_q/1000:.1f}k tok",
                    (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=11, color=INK)
    ax.set_xlabel("Tokens per question (thousands, all model calls)", color=INK2)
    ax.set_ylabel("Exact-match accuracy (%)", color=INK2)
    ax.set_ylim(-5, 110)
    ax.set_xlim(0, s.tokens_per_q.max() / 1000 * 1.6)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_title(f"HotpotQA, {meta['n_questions']} questions × {meta['repeats']} repeats · {meta['model']}",
                 color=INK, fontsize=13, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def trace_md(runs: list[dict]) -> str:
    def block(run: dict, title: str) -> str:
        lines = [f"### {title}", "", f"**Q:** {run['question']}  ", f"**Gold:** {run['gold']}  ",
                 f"**Predicted:** {run['prediction']}  ",
                 f"**Calls:** {run['calls']} · **tokens:** {run['total_tokens']:,}", "",
                 "| # | Agent | In tok | Out tok | Action |", "|---|---|---|---|---|"]
        for n, c in enumerate(run["trace"], 1):
            act = "; ".join(f"`{t['tool']}`({' '.join(t['arguments'].split())[:90]})" for t in c["tool_calls"])
            if not act:
                act = " ".join(c["text"].split())[:120].replace("|", "/")
            lines.append(f"| {n} | {c['agent']} | {c['input_tokens']:,} | {c['output_tokens']:,} | {act} |")
        return "\n".join(lines) + "\n"

    first_q = runs[0]["qid"]
    team = next(r for r in runs if r["kind"] == "team" and r["qid"] == first_q and r["repeat"] == 0)
    single = next(r for r in runs if r["kind"] == "single" and r["qid"] == first_q and r["repeat"] == 0 and r["sample"] == 0)
    return "# Example traces (repeat 0, first question)\n\n" + block(team, "Team (agents as tools)") + "\n" + block(single, "Single agent")


def main() -> None:
    df = pd.read_csv(RES / "results.csv")
    meta = json.loads((RES / "meta.json").read_text())
    runs = [json.loads(l) for l in open(RES / "runs.jsonl")]
    s = summarize(df)

    table = s.copy()
    table["accuracy_em"] = (table.accuracy_em * 100).round(0).astype(int).astype(str) + "%"
    table["acc_sd_over_repeats"] = (table.acc_sd_over_repeats * 100).round(0).astype(int).astype(str) + " pts"
    table["f1"] = table.f1.round(2)
    table["calls_per_q"] = table.calls_per_q.round(1)
    table["tokens_per_q"] = table.tokens_per_q.round(0).astype(int)
    table["tok_sd_over_repeats"] = table.tok_sd_over_repeats.round(0).astype(int)
    per_q = df.pivot_table(index=["qid", "type"], columns="condition", values="em", aggfunc="mean").round(2)
    md = (f"# Demo results\n\n`{json.dumps(meta)}`\n\n## By condition\n\n{table.to_markdown(index=False)}\n\n"
          f"## Accuracy per question (mean EM over repeats)\n\n{per_q.to_markdown()}\n")
    (RES / "summary.md").write_text(md)
    plot(s, meta, RES / "accuracy_vs_tokens.png")
    (RES / "trace_example.md").write_text(trace_md(runs))
    print(md)
    print(f"wrote {RES/'summary.md'}, {RES/'accuracy_vs_tokens.png'}, {RES/'trace_example.md'}")


if __name__ == "__main__":
    main()
