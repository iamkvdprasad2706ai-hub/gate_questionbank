from __future__ import annotations

import csv
from io import StringIO
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agents import AnswerTutorAgent, MultiAgentCoordinator, Question, SolutionStore, StudyPlannerAgent


DEFAULT_PAPERS = "/Users/kvdprasad/Documents/gatepapers"
SOLUTION_STORE = SolutionStore(Path(__file__).with_name(".gateprep.sqlite3"))

st.set_page_config(page_title="GATE DA Study Desk", page_icon="D", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    :root { --ink:#17231d; --muted:#66736b; --paper:#f6f5ef; --green:#245b48; --lime:#d6eb84; --rust:#b75837; --line:#dfe3d9; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stSidebar"] { background:#e9eee5; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { font-family:'Fraunces', serif; }
    h1, h2, h3 { color:var(--ink); }
    h1 { font-family:'Fraunces', serif; font-size:2.1rem; font-weight:500; }
    .eyebrow { font:500 .72rem 'DM Mono', monospace; text-transform:uppercase; color:var(--green); letter-spacing:0; }
    .lede { color:var(--muted); max-width:760px; }
    .metric-strip { border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:14px 0; margin:18px 0 24px; }
    div[data-testid="stMetric"] { background:transparent; border:0; padding:0; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stMetricValue"] { color:var(--green); font-family:'Fraunces',serif; }
    .source-note { border-left:3px solid var(--rust); padding:8px 12px; color:var(--muted); background:#f0eee6; font-size:.9rem; }
    div.stButton > button[kind="primary"] { background:var(--green); border:1px solid var(--green); color:white; }
    div.stButton > button { border-radius:4px; }
    code, .mono { font-family:'DM Mono',monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_questions(rows: list[dict[str, object]]) -> list[Question]:
    return [Question(**row) for row in rows]


def question_label(row: dict[str, object]) -> str:
    year = row.get("year") or "year unknown"
    return f"{year}  /  Q{row['number']}  /  {row['topic']}  /  {row['source']}"


def make_csv(rows: list[dict[str, object]], answers: dict[str, str]) -> str:
    output = StringIO()
    fields = ["year", "exam", "source", "page", "number", "marks", "topic", "text", "answer"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "answer": answers.get(str(row["question_id"]), "")})
    return output.getvalue()


def configured_api_key(entered_key: str) -> str:
    if entered_key.strip():
        return entered_key.strip()

    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key

    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except FileNotFoundError:
        return ""


if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "answers" not in st.session_state:
    st.session_state.answers = SOLUTION_STORE.load()


with st.sidebar:
    st.markdown("<div class='eyebrow'>GATE / DA</div>", unsafe_allow_html=True)
    st.title("Study desk")
    st.caption("Past-paper analysis, worked solutions, and a study plan")
    st.text_input("Paper folder", value=DEFAULT_PAPERS, key="paper_folder")
    include_non_da = st.checkbox("Include non-DA papers", value=False)
    scan = st.button("Scan paper folder", type="primary", use_container_width=True)
    st.divider()
    with st.expander("Answer model", expanded=False):
        endpoint = st.text_input(
            "OpenAI-compatible endpoint",
            value="https://api.openai.com/v1",
            help="Any compatible service works, including a local Ollama server.",
        )
        model = st.text_input("Model", value="gpt-4o-mini")
        entered_api_key = st.text_input(
            "API key override",
            type="password",
            help="Optional. Leave blank to use OPENAI_API_KEY or .streamlit/secrets.toml.",
        )
        api_key = configured_api_key(entered_api_key)
        if api_key:
            st.caption("API key configured. The key is not displayed.")
        else:
            st.caption("Add OPENAI_API_KEY to the environment or .streamlit/secrets.toml.")
        st.caption("Each generated question is sent individually to the configured model.")
    st.divider()
    st.caption("GATE score is not the same as raw marks. A target is a goal, not a prediction.")


st.markdown("<div class='eyebrow'>Question intelligence / 2024-2025</div>", unsafe_allow_html=True)
st.title("Turn past papers into a study system")
st.markdown(
    "<p class='lede'>A local multi-agent workflow reads your papers, organizes questions, surfaces recurring patterns, and builds a focused preparation schedule.</p>",
    unsafe_allow_html=True,
)

if scan:
    try:
        with st.spinner("Reader, structurer, topic analyst, and pattern analyst are working..."):
            st.session_state.dataset = MultiAgentCoordinator().process_folder(
                st.session_state.paper_folder,
                include_non_da=include_non_da,
            )
    except Exception as error:
        st.error(str(error))

dataset = st.session_state.dataset
if not dataset:
    st.markdown(
        "<div class='source-note'>Start by scanning the folder containing your PDFs. The first scan skips duplicate files and, by default, papers that do not identify as GATE DA.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

questions = dataset["questions"]
question_objects = make_questions(questions)
topics = dataset["topics"]
answers = st.session_state.answers
current_answer_count = sum(str(row["question_id"]) in answers for row in questions)

metric_columns = st.columns(4)
with metric_columns[0]:
    st.metric("Papers", len(dataset["documents"]))
with metric_columns[1]:
    st.metric("Questions", len(questions))
with metric_columns[2]:
    st.metric("Years", len({row["year"] for row in questions if row["year"]}))
with metric_columns[3]:
    st.metric("Worked answers", current_answer_count)

if dataset["warnings"]:
    with st.expander(f"Scan notes | {len(dataset['warnings'])}"):
        for warning in dataset["warnings"]:
            st.caption(warning)

overview_tab, bank_tab, patterns_tab, plan_tab = st.tabs(
    ["Overview", "Question bank", "Patterns & topics", "Preparation plan"]
)

with overview_tab:
    st.subheader("Paper set at a glance")
    st.caption("Question totals are drawn from the papers you scanned, not a forecast of future exam weightage.")
    if topics:
        topic_chart = pd.DataFrame(
            [{"Topic": row["topic"], "Observed marks": row["weighted_marks"]} for row in topics]
        )
        st.bar_chart(topic_chart, x="Observed marks", y="Topic", horizontal=True)
        st.dataframe(
            [
                {
                    "Topic": row["topic"],
                    "Questions": row["questions"],
                    "Observed marks*": row["weighted_marks"],
                    "Years found": ", ".join(row["years"]),
                }
                for row in topics
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption("*Observed marks use the mark labels in the PDF when present; otherwise each question counts once.")
    st.markdown(
        "<div class='source-note'>This collection currently contains a small number of distinct exam years. Use recurrence as a revision signal, not a guarantee about the next paper.</div>",
        unsafe_allow_html=True,
    )

with bank_tab:
    st.subheader("Question bank")
    st.caption(
        "The scanned papers do not include official answer keys. Generate worked answers "
        "with the configured model; saved answers will appear alongside each question."
    )
    years = sorted({row["year"] for row in questions if row["year"]})
    topics_all = sorted({str(row["topic"]) for row in questions})
    filter_columns = st.columns([2, 1, 2])
    chosen_topic = filter_columns[0].selectbox("Choose a topic", ["All topics"] + topics_all)
    chosen_year = filter_columns[1].selectbox("Year", ["All"] + years)
    search = filter_columns[2].text_input("Search question text")
    filtered = [
        row
        for row in questions
        if (chosen_year == "All" or row["year"] == chosen_year)
        and (chosen_topic == "All topics" or row["topic"] == chosen_topic)
        and search.casefold() in str(row["text"]).casefold()
    ]
    st.caption(f"Showing {len(filtered)} of {len(questions)} questions")

    unanswered = [row for row in filtered if str(row["question_id"]) not in answers]
    if unanswered:
        batch_columns = st.columns([2, 1])
        batch_size = batch_columns[0].number_input(
            "New solutions to generate in this run",
            min_value=1,
            max_value=min(20, len(unanswered)),
            value=min(5, len(unanswered)),
            key="topic_batch_size",
        )
        if batch_columns[1].button(
            "Generate solutions",
            type="primary",
            key="topic_batch_generate",
            use_container_width=True,
        ):
            progress = st.progress(0)
            status = st.empty()
            tutor = AnswerTutorAgent()
            batch = unanswered[: int(batch_size)]
            for index, row in enumerate(batch, start=1):
                status.caption(f"Solving {index} of {len(batch)} | {question_label(row)}")
                try:
                    answer = tutor.answer(Question(**row), endpoint, model, api_key)
                except Exception as error:
                    st.error(f"Stopped at {question_label(row)}: {error}")
                    break
                question_id = str(row["question_id"])
                SOLUTION_STORE.save(question_id, answer)
                st.session_state.answers[question_id] = answer
                progress.progress(index / len(batch))
            status.caption("Generated answers are saved locally and remain available after restart.")
            st.rerun()

    for row in filtered:
        question_id = str(row["question_id"])
        st.markdown(
            f"#### {row['year'] or 'Year unknown'} · Q{row['number']} · "
            f"{row['marks'] or '?'} marks"
        )
        st.caption(f"{row['topic']} · {row['source']} · page {row['page']}")
        st.markdown("**Question**")
        st.markdown(str(row["text"]))

        if question_id in answers:
            st.markdown("**Answer and worked solution**")
            st.caption("AI-prepared solution. Verify calculations and assumptions before relying on it.")
            st.markdown(answers[question_id])
        else:
            st.info("No worked answer saved yet.")
            if st.button("Generate answer and solution", key=f"solve_{question_id}"):
                try:
                    with st.spinner("Tutor agent is solving this question..."):
                        answer = AnswerTutorAgent().answer(
                            Question(**row), endpoint, model, api_key
                        )
                        SOLUTION_STORE.save(question_id, answer)
                        st.session_state.answers[question_id] = answer
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
        st.divider()

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download question bank · CSV",
        make_csv(questions, answers),
        file_name="gate_da_question_bank.csv",
        mime="text/csv",
    )
    export_rows = [
        {**row, "answer": answers.get(str(row["question_id"]), "")}
        for row in questions
    ]
    download_columns[1].download_button(
        "Download question bank · JSON",
        json.dumps(export_rows, ensure_ascii=False, indent=2),
        file_name="gate_da_question_bank.json",
        mime="application/json",
    )
    if answers:
        solved = [
            {**row, "answer": answers.get(str(row["question_id"]), "")}
            for row in questions
            if str(row["question_id"]) in answers
        ]
        download_columns[2].download_button(
            "Download worked solutions · JSON",
            json.dumps(solved, ensure_ascii=False, indent=2),
            file_name="gate_da_worked_solutions.json",
            mime="application/json",
        )

with patterns_tab:
    st.subheader("Recurring question patterns")
    st.caption("Similarity is a text-based candidate score. Open each pair and verify that the underlying concept really matches.")
    minimum_similarity = st.slider("Minimum text similarity", 0.28, 0.85, 0.32, 0.01)
    candidates = [
        pair for pair in dataset["similar"] if pair["similarity"] >= minimum_similarity
    ]
    if not candidates:
        st.info("No cross-paper pairs meet this threshold. Lower it slightly to inspect weaker text matches.")
    for pair in candidates:
        left = pair["left"]
        right = pair["right"]
        label = f"{left.year} Q{left.number} <-> {right.year} Q{right.number} | {pair['similarity']:.0%} text match"
        with st.expander(label):
            st.markdown(f"**{left.topic}** | {left.source}, page {left.page}")
            st.write(left.text)
            st.markdown(f"**{right.topic}** | {right.source}, page {right.page}")
            st.write(right.text)
    st.divider()
    st.subheader("Topic evidence")
    st.dataframe(
        [
            {
                "Topic": row["topic"],
                "Questions": row["questions"],
                "Observed marks": row["weighted_marks"],
                "Years": ", ".join(row["years"]),
                "Examples": "; ".join(row["examples"]),
            }
            for row in topics
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Topic assignment is keyword-based. Review 'Unclassified' and borderline questions manually.")

with plan_tab:
    st.subheader("Preparation plan")
    controls = st.columns(3)
    duration = controls[0].slider("Weeks until exam", min_value=4, max_value=24, value=12)
    weekly_hours = controls[1].number_input("Study hours per week", min_value=1, max_value=80, value=12)
    target_score = controls[2].number_input("Target GATE score", min_value=1, max_value=1000, value=800)
    plan = StudyPlannerAgent().build_plan(
        question_objects,
        weeks=duration,
        hours_per_week=int(weekly_hours),
        target_score=int(target_score),
    )
    st.markdown(
        f"<div class='source-note'>Target: {plan['target_score']} / 1000 GATE score. {plan['evidence_note']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Focus first, based on the observed paper set**")
    for row in plan["topics"]:
        st.markdown(
            f"- **{row['topic']}**: {row['questions']} questions, "
            f"{row['weighted_marks']} observed marks across {', '.join(row['years']) or 'unknown years'}"
        )
    for week in plan["weeks"]:
        with st.expander(f"Week {week['week']:02d} · {week['title']} · {week['hours']} h"):
            for action in week["actions"]:
                st.markdown(f"- {action}")
    st.download_button(
        "Download preparation plan · JSON",
        json.dumps(plan, ensure_ascii=False, indent=2),
        file_name="gate_da_preparation_plan.json",
        mime="application/json",
    )
