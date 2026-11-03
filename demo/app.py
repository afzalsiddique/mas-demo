"""Side-by-side demo UI: pick a question (or type your own) and two setups, press Run.

The left setup runs first and its result appears all at once, then the right one.
Start it with `qsub -v MODE=ui run_demo.pbs` (vLLM + this app on one GPU node) and
open the SSH tunnel printed in logs/ui_tunnel.txt.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import streamlit as st

import mas_demo as md

st.set_page_config(page_title="Single agent vs. team", layout="wide")

QUESTIONS = md.load_questions()
CUSTOM = len(QUESTIONS)  # index of the "type your own" entry in the question list
try:
    DEFAULT_K = json.loads((md.HERE / "results" / "meta.json").read_text())["k"]
except Exception:
    DEFAULT_K = 4
SETUP_KEYS = list(md.SETUPS)  # single, vote, team


@st.cache_resource(show_spinner="Indexing the 66,581 Wikipedia paragraphs (once) …")
def corpus() -> md.CorpusStore:
    return md.get_corpus()


def setup_label(key: str, k: int | None = None) -> str:
    return md.SETUPS[key].replace("× k", f"× {k}") if k else md.SETUPS[key]


def question_label(i: int) -> str:
    if i == CUSTOM:
        return "✏️ Type your own question (searches 66,581 Wikipedia paragraphs)"
    q = QUESTIONS[i]
    tag = "" if q.get("experiment", True) else " (extra)"
    return f"{i + 1}. [{q['type']}{tag}] {q['question']}"


# ---------------------------------------------------------------- controls
st.title("One agent or a team?")
st.caption(f"Same model ({md.MODEL_NAME}), same search tool, same temperature ({md.TEMPERATURE}). "
           "Only the organization changes. Every model call is counted, including the specialists' calls.")

qi = st.selectbox("Question", range(len(QUESTIONS) + 1), format_func=question_label,
                  help="1-3 are hand-picked for the demo script (docs/demo-script.md); 4-8 are the questions behind the "
                       "pre-computed experiment results; 9-23 are extra HotpotQA questions. "
                       "Each HotpotQA question searches only its own 10 paragraphs.")

if qi == CUSTOM:
    cq1, cq2 = st.columns([3, 1])
    custom_q = cq1.text_input("Your question", placeholder="e.g. In what year was the director of The Prestige born?")
    custom_gold = cq2.text_input("Gold answer (for ✅/❌)", placeholder="e.g. 1970")
    st.caption("The agents search 66,581 local Wikipedia paragraphs ")
    q = {"qid": "custom", "type": "custom", "question": custom_q.strip(), "answer": custom_gold.strip(),
         "docs": None}
else:
    q = QUESTIONS[qi]

c1, c2, c3 = st.columns([2, 2, 1])
left = c1.selectbox("Left", SETUP_KEYS, index=0, format_func=setup_label)
right = c2.selectbox("Right", SETUP_KEYS, index=2, format_func=setup_label)
uses_vote = "vote" in (left, right)
k = int(c3.number_input(
    "k (tries for the vote setup)", min_value=2, max_value=10, value=DEFAULT_K, disabled=not uses_vote,
    help="Only the 'Single agent × k tries, majority vote' setup uses k: it runs the single agent k times "
         "and takes the most common answer, so its cost is about k × one try. "
         f"{DEFAULT_K} ≈ the team's token budget in our experiment. "
         "k has no effect on 'Single agent (1 try)' or on the team."))
if not uses_vote:
    c3.caption("k is off: pick the vote setup on one side to use it.")

with st.expander("Display options", expanded=False):
    o1, o2, o3, o4 = st.columns(4)
    show_table = o1.toggle("Call table", value=True)
    show_io = o2.toggle("Input & output of every call", value=True)
    show_system = o3.toggle("Show system prompts", value=False, disabled=not show_io,
                            help="Each agent's fixed instructions. Hidden by default because they repeat.")
    show_repeat = o4.toggle("Repeat earlier messages", value=False, disabled=not show_io,
                            help="Each call re-sends the whole conversation of that agent so far. "
                                 "Off: show only what is new in each call's input.")
    expand_all = o1.toggle("Expand every call", value=False, disabled=not show_io)

ready = bool(q["question"])
run = st.button("Run ▶", type="primary", disabled=not ready)
if not ready:
    st.info("Type a question first.")


# ---------------------------------------------------------------- rendering
def tool_call_str(t: dict) -> str:
    return f"{t['tool']}({t['arguments']})"


def render_call(n: int, c: dict, calls: list[dict], is_vote: bool) -> None:
    head = f"{n}. {c['agent']}" + (f" · try {c['attempt']}" if is_vote else "") + \
        f" — {c['input_tokens']:,} in / {c['output_tokens']:,} out tokens"
    with st.expander(head, expanded=expand_all):
        msgs = c.get("messages") or []
        # find the earlier call of the same agent whose input is a prefix of this one
        start, prev = 0, None
        if not show_repeat:
            for j in range(n - 2, -1, -1):
                p = calls[j]
                pm = p.get("messages") or []
                if p["agent"] == c["agent"] and p.get("attempt") == c.get("attempt") \
                        and 0 < len(pm) < len(msgs) and msgs[:len(pm)] == pm:
                    start, prev = len(pm), j + 1
                    break
        st.markdown("**▶ Input sent to the model**")
        if c.get("tools_offered"):
            st.caption("Tools offered: " + "; ".join(f"`{t['name']}`: {t['description']}" for t in c["tools_offered"]))
        if prev:
            st.caption(f"(The first {start} messages are the same as the input of call {prev}; "
                       "turn on 'Repeat earlier messages' to see them.)")
        for m in msgs[start:]:
            if m["role"] == "system" and not show_system:
                st.caption(f"system prompt: {len(m['text']):,} characters (hidden; see Display options)")
                continue
            st.markdown(f"`{m['role']}`")
            st.code(m["text"] or "(empty)", language=None, wrap_lines=True)
        st.markdown("**◀ Output from the model**")
        for t in c["tool_calls"]:
            st.code("→ calls " + tool_call_str(t), language=None, wrap_lines=True)
        if c["text"]:
            st.code(c["text"], language=None, wrap_lines=True)
        if not c["tool_calls"] and not c["text"]:
            st.caption("(empty output)")


def render(col, res: dict | None, key: str) -> None:
    with col:
        st.subheader(setup_label(key, k))
        if res is None:
            st.info("Not run yet.")
            return
        if q["answer"]:
            mark = "✅ correct" if res["correct"] else "❌ wrong"
            st.markdown(f"### Answer: {res['answer'] or '—'}  \n{mark} (gold: *{q['answer']}*)")
        else:
            st.markdown(f"### Answer: {res['answer'] or '—'}  \n(no gold answer given)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model calls", res["n_calls"])
        m2.metric("Tokens", f"{res['tokens']:,}")
        m3.metric("Seconds", res["seconds"])
        is_vote = res["setup"] == "vote"
        if is_vote:
            st.markdown("**Answers per try:** " + " · ".join(f"`{a or '—'}`" for a in res["attempt_answers"]))
        for e in res["errors"]:
            st.warning(f"Run error (counted as no answer): {e}")
        if show_table:
            rows = []
            for n, c in enumerate(res["calls"], 1):
                action = "; ".join(tool_call_str(t) for t in c["tool_calls"]) or c["text"]
                row = {"#": n, "Agent": c["agent"], "In tok": c["input_tokens"], "Out tok": c["output_tokens"],
                       "Action": md._short(action, 110)}
                rows.append({"Try": c["attempt"], **row} if is_vote else row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if show_io:
            st.markdown("**Every call, in order** (click one to open it)")
            for n, c in enumerate(res["calls"], 1):
                render_call(n, c, res["calls"], is_vote)
        st.download_button("Download this run (JSON)", json.dumps(res, indent=2, ensure_ascii=False),
                           file_name=f"{key}_{q['qid']}.json", mime="application/json", key=f"dl_{col_id(col)}")


def col_id(col) -> str:
    return "L" if col is L else "R"


state = st.session_state
sig = (qi, q["question"], q["answer"], left, right, k if uses_vote else None)
if state.get("sig") != sig:  # changing a question or setup clears old results
    state.sig, state.left_res, state.right_res = sig, None, None

if q.get("docs") is None:
    corpus()  # build the index before the first custom run (cached afterwards)

L, R = st.columns(2, gap="large")
if run:
    state.left_res = state.right_res = None
    render(R, None, right)
    with L:
        with st.spinner(f"Running {setup_label(left, k)} …"):
            state.left_res = asyncio.run(md.run_setup(left, q, k))
    render(L, state.left_res, left)
    with R:
        with st.spinner(f"Running {setup_label(right, k)} …"):
            state.right_res = asyncio.run(md.run_setup(right, q, k))
    st.rerun()  # redraw both panels cleanly
else:
    render(L, state.left_res, left)
    render(R, state.right_res, right)

lr, rr = state.left_res, state.right_res
if lr and rr and lr["tokens"]:
    st.divider()
    line = (f"**Right vs. left:** {rr['n_calls'] / max(1, lr['n_calls']):.1f}× the calls, "
            f"{rr['tokens'] / lr['tokens']:.1f}× the tokens, {rr['seconds'] / max(0.1, lr['seconds']):.1f}× the time.")
    if q["answer"]:
        line += f" Correct: left {'✅' if lr['correct'] else '❌'} · right {'✅' if rr['correct'] else '❌'}"
    st.markdown(line)
    st.caption("Temperature 0.7: a single run can differ from the pre-computed results in results/ "
               "(5 repeats per question). Input tokens also include the tool definitions, which the "
               "server adds to the prompt and which are not shown above.")
