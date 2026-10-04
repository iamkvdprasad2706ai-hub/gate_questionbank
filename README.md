# GATE DA Study Desk

A local Python workbench for extracting GATE Data Science and Artificial Intelligence questions from past-paper PDFs, reviewing recurring patterns, generating worked solutions, and building a preparation schedule.

## Agents

- **Paper reader** scans a folder recursively, extracts selectable PDF text, records page/source details, skips byte-identical duplicates, and flags scanned PDFs that need OCR.
- **Question structurer** separates numbered questions, carries page and mark metadata, and assigns an initial syllabus topic label.
- **Topic analyst** summarizes question and observed-mark counts by topic and year.
- **Pattern analyst** ranks cross-paper text matches with a local TF-IDF cosine score. These are candidates for human review, not confirmed conceptual repeats.
- **Tutor** creates a step-by-step worked answer through an OpenAI-compatible model endpoint. Each question is sent separately; answers are saved in a local SQLite database and can be exported.
- **Study planner** uses the scanned topic evidence and a student's available weeks/hours to produce a week-by-week schedule.
- **Coordinator** runs the local reading and analysis agents as one folder-processing workflow.

## Run locally

From this project folder, create and activate a virtual environment, then install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The default paper folder is `/Users/kvdprasad/Documents/gatepapers`. Use **Scan paper folder** in the sidebar. GATE DA PDFs are included by default; the CS papers in that folder are skipped unless **Include non-DA papers** is enabled. Duplicate DA24S1 copies are detected by content hash and counted once.

## Worked solutions

In **Answer model**, the default endpoint is OpenAI and the default model is `gpt-4o-mini`. Configure the API key server-side using either option below; `.streamlit/secrets.toml` is already ignored by Git:

```toml
OPENAI_API_KEY = "your-api-key"
```

Alternatively, set `OPENAI_API_KEY` in the environment before starting Streamlit. The app checks the masked sidebar override first, then the environment variable, then `.streamlit/secrets.toml`; configured keys are not shown in the UI. Use the question-level action to solve one question or the batch action to solve up to 20 unanswered questions at a time. Answers are stored in `.gateprep.sqlite3` in this project folder, so batch progress survives app restarts. Review generated answers before relying on them, especially when a PDF extraction is incomplete or a question depends on a diagram.

The project does not include an OCR engine. Papers with no selectable text are called out in scan notes. Topic labels are keyword-based and can be imperfect; inspect the **Unclassified** topic and the original question before treating a topic total as definitive.

## Score target and evidence limits

The preparation plan accepts a target GATE score (default 800), available weeks, and weekly study hours. It is a planning target, not a score prediction or guarantee. Current topic recurrence is based only on the distinct years available in the scanned folder; past-paper frequency cannot guarantee future exam weightage. GATE score and raw marks are different measures.

## Tests

```sh
python -m unittest discover -s tests
```