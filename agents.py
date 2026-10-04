from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader


@dataclass
class PaperDocument:
    source: str
    digest: str
    title: str
    exam: str
    year: int | None
    page_count: int
    pages: list[str] = field(repr=False)


@dataclass
class Question:
    question_id: str
    source: str
    year: int | None
    exam: str
    page: int
    number: int
    marks: int | None
    topic: str
    text: str


TOPIC_PATTERNS = {
    "Probability & Statistics": re.compile(
        r"\b(probabilit\w*|random variable|distribution|expected value|variance|"
        r"standard deviation|bayes|poisson|normal|binomial|covariance|correlation|"
        r"likelihood|estimat\w*|hypothesis|posterior|prior)\b",
        re.I,
    ),
    "Linear Algebra": re.compile(
        r"\b(matrix|matrices|eigenvalue|eigenvector|determinant|rank|basis|"
        r"subspace|orthogonal|vector|diagonal|inverse|span)\b",
        re.I,
    ),
    "Calculus & Optimization": re.compile(
        r"\b(derivative|gradient|hessian|integral|maximum|minimum|minimi[sz]\w*|"
        r"maximi[sz]\w*|convex|lagrange|jacobian|partial derivative)\b",
        re.I,
    ),
    "Programming & DSA": re.compile(
        r"\b(python|pseudocode|algorithm|time complexity|space complexity|"
        r"binary tree|graph|sorting|stack|queue|recursion|linked list|hash table|"
        r"array|loop|program|code snippet)\b",
        re.I,
    ),
    "DBMS & Warehousing": re.compile(
        r"\b(sql|database|relational|normal form|schema|data warehouse|join|"
        r"primary key|foreign key|transaction|index|entity relationship)\b",
        re.I,
    ),
    "Machine Learning": re.compile(
        r"\b(regression|classif\w*|overfit\w*|regulari[sz]\w*|gradient descent|"
        r"k-means|clustering|support vector|\bsvm\b|decision tree|random forest|"
        r"feature|neural network|loss function|training data|test data)\b",
        re.I,
    ),
    "Artificial Intelligence": re.compile(
        r"\b(search tree|heuristic|\ba\*\b|minimax|propositional|predicate logic|"
        r"knowledge representation|artificial intelligence|intelligent agent|"
        r"state space|inference|adversarial)\b",
        re.I,
    ),
}

TOPIC_FOCUS_AREAS = {
    "General Aptitude": ["verbal reasoning", "numerical aptitude", "analytical reasoning"],
    "Probability & Statistics": [
        "counting and conditional probability",
        "random variables and standard distributions",
        "expectation, variance, covariance, and correlation",
        "sampling, estimation, confidence intervals, and hypothesis tests",
    ],
    "Linear Algebra": [
        "systems of linear equations, rank, and determinants",
        "vector spaces, independence, basis, and dimension",
        "eigenvalues, eigenvectors, and diagonalization",
        "orthogonality, projections, and least squares",
    ],
    "Calculus & Optimization": [
        "limits, continuity, and single-variable derivatives",
        "partial derivatives, gradients, Jacobians, and Hessians",
        "unconstrained extrema and convexity",
        "constrained optimization and Lagrange multipliers",
    ],
    "Programming & DSA": [
        "Python expressions, control flow, functions, and data types",
        "arrays, stacks, queues, trees, and graph representations",
        "searching, sorting, hashing, and graph traversal",
        "time and space complexity, recursion, and dynamic programming",
    ],
    "DBMS & Warehousing": [
        "ER models, relational algebra, and relational schemas",
        "keys, functional dependencies, and normalization",
        "SQL queries, joins, and aggregation",
        "transactions, indexing, warehouse schemas, and OLAP",
    ],
    "Machine Learning": [
        "linear and logistic regression, and classification basics",
        "decision trees, SVM, k-nearest neighbours, and Naive Bayes",
        "regularization, bias-variance, and model validation",
        "clustering, dimensionality reduction, and evaluation metrics",
    ],
    "Artificial Intelligence": [
        "state-space search and uninformed search strategies",
        "heuristic search and constraint-satisfaction problems",
        "adversarial search and minimax reasoning",
        "propositional logic, inference, and knowledge representation",
    ],
    "Unclassified": ["review each source question and assign its syllabus area manually"],
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")
STOP_WORDS = set(
    "a an and are as at be been being by can could did do does for from had has "
    "have he her here hers him his how i if in into is it its may might more most "
    "must my no not of on one or our ours she should so than that the their theirs "
    "them then there these they this those through to two under up us was we were "
    "what when where which who why will with would you your yours given following "
    "select appropriate answer following above below find determine calculate".split()
)


class PaperReaderAgent:
    """Find readable exam PDFs and ignore byte-identical copies."""

    def read_folder(
        self, folder: str | Path, include_non_da: bool = False
    ) -> tuple[list[PaperDocument], list[str]]:
        root = Path(folder).expanduser()
        if not root.is_dir():
            raise ValueError(f"Paper folder does not exist: {root}")

        documents: list[PaperDocument] = []
        warnings: list[str] = []
        seen: set[str] = set()
        for path in sorted(root.rglob("*.pdf")):
            try:
                file_bytes = path.read_bytes()
                digest = sha256(file_bytes).hexdigest()
                if digest in seen:
                    warnings.append(f"Skipped duplicate PDF: {path.name}")
                    continue

                reader = PdfReader(path)
                pages = [page.extract_text() or "" for page in reader.pages]
                if not any(page.strip() for page in pages):
                    warnings.append(f"No selectable text found in {path.name}; OCR is not enabled.")
                    continue

                title = next(
                    (line.strip() for page in pages for line in page.splitlines() if line.strip()),
                    path.stem,
                )
                exam_text = f"{path.name} {title} {' '.join(pages[:2])}".lower()
                is_da = bool(re.search(r"\bda\b|data science|artificial intelligence", exam_text))
                if not is_da and not include_non_da:
                    warnings.append(f"Skipped non-DA paper: {path.name}")
                    continue

                year_match = re.search(r"20\d{2}", f"{path.name} {title}")
                short_year_match = re.search(r"\bDA(\d{2})S\d+\b", path.name, re.I)
                documents.append(
                    PaperDocument(
                        source=str(path.relative_to(root)),
                        digest=digest,
                        title=title,
                        exam="DA" if is_da else "Other",
                        year=(
                            int(year_match.group())
                            if year_match
                            else 2000 + int(short_year_match.group(1))
                            if short_year_match
                            else None
                        ),
                        page_count=len(reader.pages),
                        pages=pages,
                    )
                )
                seen.add(digest)
            except Exception as error:
                warnings.append(f"Could not read {path.name}: {error}")
        return documents, warnings


class QuestionStructuringAgent:
    """Split extracted exam text into numbered, page-linked questions."""

    question_start = re.compile(r"^\s*Q\s*\.?\s*(\d{1,2})\s*[).:]?\s*(.*)$", re.I)
    section_header = re.compile(
        r"^\s*Q\s*\.?\s*(\d+)\s*[–—-]\s*Q\s*\.?\s*(\d+).*?\b(ONE|TWO)\s+marks?\b",
        re.I,
    )
    header_range = re.compile(r"^\s*Q\s*\.?\s*\d+\s*[–—-]\s*Q\s*\.?\s*\d+", re.I)

    def structure(self, document: PaperDocument) -> list[Question]:
        starts: list[tuple[int, int, str]] = []
        marks_by_number: dict[int, int] = {}
        current_marks: int | None = None
        for page_number, page_text in enumerate(document.pages, start=1):
            for line_index, line in enumerate(page_text.splitlines()):
                section = self.section_header.match(line)
                if section:
                    current_marks = 1 if section.group(3).upper() == "ONE" else 2
                    for number in range(int(section.group(1)), int(section.group(2)) + 1):
                        marks_by_number[number] = current_marks
                    continue

                match = self.question_start.match(line)
                if not match or self.header_range.match(line):
                    continue
                number = int(match.group(1))
                marks_by_number.setdefault(number, current_marks or 0)
                starts.append((page_number, line_index, line))

        questions: list[Question] = []
        for index, (page_number, line_index, first_line) in enumerate(starts):
            page_text = document.pages[page_number - 1].splitlines()
            end_page = starts[index + 1][0] if index + 1 < len(starts) else len(document.pages)
            end_offset = (
                starts[index + 1][1]
                if index + 1 < len(starts)
                else len(document.pages[end_page - 1].splitlines())
            )

            if index + 1 < len(starts) and starts[index + 1][0] == page_number:
                block_lines = page_text[line_index:end_offset]
            else:
                block_lines = page_text[line_index:]
                for page_index in range(page_number, end_page - 1):
                    block_lines.extend(document.pages[page_index].splitlines())
                if index + 1 < len(starts):
                    block_lines.extend(document.pages[end_page - 1].splitlines()[:end_offset])

            text = self._clean_text("\n".join(block_lines))
            match = self.question_start.match(first_line)
            if not match or len(text) < 12:
                continue
            number = int(match.group(1))
            question_id = f"{document.digest[:10]}-{number:02d}-{page_number}"
            questions.append(
                Question(
                    question_id=question_id,
                    source=document.source,
                    year=document.year,
                    exam=document.exam,
                    page=page_number,
                    number=number,
                    marks=marks_by_number.get(number) or None,
                    topic=self.classify_topic(number, text),
                    text=text,
                )
            )
        return questions

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r"\s+", " ", line).strip()
            if not cleaned or re.match(r"Page\s+\d+\s+of\s+\d+", cleaned, re.I):
                continue
            if QuestionStructuringAgent.header_range.match(cleaned):
                continue
            lines.append(cleaned)
        return "\n".join(lines)

    @staticmethod
    def classify_topic(number: int, text: str) -> str:
        if number <= 10:
            return "General Aptitude"
        for topic, pattern in TOPIC_PATTERNS.items():
            if pattern.search(text):
                return topic
        return "Unclassified"


class SimilarityAgent:
    """Rank possible repeats with a transparent local TF-IDF cosine score."""

    def find_similar(
        self, questions: list[Question], threshold: float = 0.28, limit: int = 100
    ) -> list[dict[str, object]]:
        tokenized = [self._tokens(question.text) for question in questions]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        document_count = max(len(questions), 1)

        vectors: list[dict[str, float]] = []
        for tokens in tokenized:
            counts = Counter(tokens)
            vector = {
                token: (1 + math.log(count))
                * (math.log((document_count + 1) / (document_frequency[token] + 1)) + 1)
                for token, count in counts.items()
            }
            vectors.append(vector)

        norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in vectors]
        results = []
        for left_index, right_index in combinations(range(len(questions)), 2):
            if questions[left_index].source == questions[right_index].source:
                continue
            left, right = vectors[left_index], vectors[right_index]
            if not norms[left_index] or not norms[right_index]:
                continue
            shared = left.keys() & right.keys()
            score = sum(left[token] * right[token] for token in shared) / (
                norms[left_index] * norms[right_index]
            )
            if score >= threshold:
                results.append(
                    {
                        "left": questions[left_index],
                        "right": questions[right_index],
                        "similarity": round(score, 3),
                        "same_topic": questions[left_index].topic == questions[right_index].topic,
                    }
                )
        return sorted(results, key=lambda item: item["similarity"], reverse=True)[:limit]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if token.lower() not in STOP_WORDS and not token.isdigit()
        ]


class AnswerTutorAgent:
    """Generate a worked solution through an OpenAI-compatible chat endpoint."""

    def answer(
        self,
        question: Question,
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout: int = 120,
    ) -> str:
        endpoint = endpoint.strip().rstrip("/")
        model = model.strip()
        if not endpoint or not model:
            raise ValueError("Enter a model endpoint and model name to generate solutions.")

        url = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"
        prompt = (
            "Solve this GATE Data Science and Artificial Intelligence question carefully. "
            "Show the governing concept, step-by-step derivation or reasoning, the final "
            "answer, a quick verification, and one common exam trap. Preserve units and "
            "state any assumptions. If the extracted text is incomplete or ambiguous, "
            "say exactly what is missing instead of guessing.\n\n"
            f"Paper: {question.exam} {question.year or ''}; question {question.number}; "
            f"{question.marks or 'unknown'} marks.\n"
            f"Question text:\n{question.text}"
        )
        payload = json.dumps(
            {
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are a precise GATE DA tutor. Do not invent unreadable question details."},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        request = Request(url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Model endpoint returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach the model endpoint: {error.reason}") from error
        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("The model endpoint returned an unexpected response.") from error


class SolutionStore:
    """Keep generated answers on disk without persisting model credentials."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS solutions ("
                "question_id TEXT PRIMARY KEY, answer TEXT NOT NULL, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def load(self) -> dict[str, str]:
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute("SELECT question_id, answer FROM solutions").fetchall()
        return {question_id: answer for question_id, answer in rows}

    def save(self, question_id: str, answer: str) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO solutions (question_id, answer) VALUES (?, ?) "
                "ON CONFLICT(question_id) DO UPDATE SET "
                "answer = excluded.answer, updated_at = CURRENT_TIMESTAMP",
                (question_id, answer),
            )


class TopicAnalysisAgent:
    def summarize(self, questions: list[Question]) -> list[dict[str, object]]:
        counts: dict[str, Counter[int | None]] = defaultdict(Counter)
        examples: dict[str, list[str]] = defaultdict(list)
        for question in questions:
            counts[question.topic][question.year] += question.marks or 1
            if len(examples[question.topic]) < 3:
                examples[question.topic].append(
                    f"{question.year or 'Year unknown'} Q{question.number} ({question.source})"
                )

        result = []
        for topic, years in counts.items():
            result.append(
                {
                    "topic": topic,
                    "questions": sum(1 for question in questions if question.topic == topic),
                    "weighted_marks": sum(years.values()),
                    "years": sorted(str(year) for year, count in years.items() if count),
                    "examples": examples[topic],
                    "focus_areas": TOPIC_FOCUS_AREAS.get(topic, TOPIC_FOCUS_AREAS["Unclassified"]),
                }
            )
        return sorted(result, key=lambda row: (row["weighted_marks"], row["questions"]), reverse=True)


class StudyPlannerAgent:
    def build_plan(
        self,
        questions: list[Question],
        weeks: int = 12,
        hours_per_week: int = 12,
        target_score: int = 800,
    ) -> dict[str, object]:
        if not questions:
            return {"target_score": target_score, "weeks": [], "topics": []}

        topic_rows = TopicAnalysisAgent().summarize(questions)
        ranked_topics = [
            row["topic"]
            for row in topic_rows
            if row["topic"] not in {"General Aptitude", "Unclassified"}
        ]
        topic_evidence = {row["topic"]: row for row in topic_rows}
        topic_slots = max(1, weeks - 4)
        topic_weeks: dict[int, list[str]] = defaultdict(list)
        for index, topic in enumerate(ranked_topics):
            week_number = 3 + (index % topic_slots)
            topic_weeks[week_number].append(topic)

        plan_weeks = []
        for week in range(1, weeks + 1):
            if week == 1:
                title = "Baseline and exam map"
                actions = [
                    "Take a timed diagnostic from an untouched paper and record accuracy by topic.",
                    "Create an error log: concept gap, calculation slip, time pressure, or misread question.",
                    "Revise General Aptitude foundations and set a weekly review slot.",
                ]
            elif week == 2:
                title = "Mathematical foundations"
                actions = [
                    "Review probability, statistics, linear algebra, calculus, and optimization prerequisites.",
                    "Build a one-page formula sheet from memory, then verify it against worked examples.",
                    "Solve a short mixed set under time limits and update the error log.",
                ]
            elif week in topic_weeks:
                assigned = topic_weeks[week]
                title = " + ".join(assigned) if assigned else "Core-topic consolidation"
                actions = []
                for topic in assigned:
                    evidence = topic_evidence[topic]
                    focus = "; ".join(evidence["focus_areas"])
                    years_seen = ", ".join(evidence["years"]) or "year unknown"
                    actions.append(f"{topic}: study {focus}; explain each idea without notes.")
                    actions.append(
                        f"Practice the {evidence['questions']} tagged past-paper questions "
                        f"from {years_seen}; mark each attempt and review every miss."
                    )
                actions.extend(
                    [
                        "Solve the matching past-paper questions, then redo missed questions after 48 hours.",
                        "Record accuracy, time per mark, and the error type for every attempt.",
                    ]
                )
            elif week == weeks - 1:
                title = "Final review and exam routine"
                actions = [
                    "Review the error log and formula sheet; do not add large new topics.",
                    "Take one final timed mock, analyze it, and protect sleep and exam-day logistics.",
                    "Prioritize accuracy and question selection over attempting every question.",
                ]
            else:
                title = "Timed mixed practice and repair"
                actions = [
                    "Attempt one timed mixed section or full paper and review every uncertain answer.",
                    "Revisit the two lowest-accuracy topics from your own tracker.",
                    "Repeat previously missed questions without looking at the worked solution.",
                ]
            plan_weeks.append(
                {
                    "week": week,
                    "title": title,
                    "hours": hours_per_week,
                    "actions": actions,
                }
            )

        years = sorted({question.year for question in questions if question.year})
        return {
            "target_score": target_score,
            "weeks": plan_weeks,
            "topics": topic_rows,
            "paper_years": years,
            "evidence_note": (
                f"This plan is based on {len(questions)} extracted questions from "
                f"{len(years)} distinct exam year(s). Topic labels are keyword-based; "
                "a small past-paper sample cannot predict the next paper or guarantee a score."
            ),
        }


class MultiAgentCoordinator:
    """Coordinate independent reader, structuring, analysis, and planning agents."""

    def __init__(self) -> None:
        self.reader = PaperReaderAgent()
        self.structurer = QuestionStructuringAgent()
        self.similarity = SimilarityAgent()
        self.topic_analysis = TopicAnalysisAgent()
        self.planner = StudyPlannerAgent()

    def process_folder(
        self, folder: str | Path, include_non_da: bool = False
    ) -> dict[str, object]:
        documents, warnings = self.reader.read_folder(folder, include_non_da)
        questions = [
            question
            for document in documents
            for question in self.structurer.structure(document)
        ]
        return {
            "documents": [asdict(document) | {"pages": None} for document in documents],
            "questions": [asdict(question) for question in questions],
            "warnings": warnings,
            "topics": self.topic_analysis.summarize(questions),
            "similar": self.similarity.find_similar(questions),
        }