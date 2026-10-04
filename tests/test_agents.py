import unittest
from tempfile import TemporaryDirectory

from agents import (
    PaperDocument,
    Question,
    QuestionStructuringAgent,
    SimilarityAgent,
    SolutionStore,
    StudyPlannerAgent,
    TopicAnalysisAgent,
)


class QuestionStructuringTests(unittest.TestCase):
    def test_sections_question_marks_and_topic_labels(self) -> None:
        document = PaperDocument(
            source="DA2025.pdf",
            digest="0123456789abcdef",
            title="Data Science and Artificial Intelligence",
            exam="DA",
            year=2025,
            page_count=1,
            pages=[
                "Q.1 – Q.2 Carry ONE mark Each\n"
                "Q.1 Find the probability of drawing a red ball.\n"
                "(A) 1/2\n(B) 1/3\n"
                "Q.2 Compute the eigenvalue of matrix A.\n"
                "Q.3 – Q.3 Carry TWO marks Each\n"
                "Q.3 Find the gradient of the objective function."
            ],
        )

        questions = QuestionStructuringAgent().structure(document)

        self.assertEqual([question.number for question in questions], [1, 2, 3])
        self.assertEqual([question.marks for question in questions], [1, 1, 2])
        self.assertEqual(
            [question.topic for question in questions],
            ["General Aptitude", "General Aptitude", "General Aptitude"],
        )
        self.assertNotIn("Carry ONE mark", questions[0].text)

    def test_content_question_is_classified_by_concept(self) -> None:
        topic = QuestionStructuringAgent.classify_topic(
            18, "A random variable follows a normal distribution with known variance."
        )

        self.assertEqual(topic, "Probability & Statistics")


class AnalysisAgentTests(unittest.TestCase):
    def test_similarity_finds_cross_paper_wording_reuse(self) -> None:
        first = Question(
            "a-11-1", "DA2024.pdf", 2024, "DA", 12, 11, 2,
            "Probability & Statistics", "Find the expected value of this random variable distribution.",
        )
        second = Question(
            "b-12-1", "DA2025.pdf", 2025, "DA", 13, 12, 2,
            "Probability & Statistics", "Determine the expected value for the random variable distribution.",
        )

        matches = SimilarityAgent().find_similar([first, second], threshold=0.15)

        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["same_topic"])

    def test_topic_summary_and_plan_include_year_evidence(self) -> None:
        questions = [
            Question("a", "DA2024.pdf", 2024, "DA", 12, 11, 2,
                     "Linear Algebra", "Find the eigenvalue of a matrix."),
            Question("b", "DA2025.pdf", 2025, "DA", 12, 11, 2,
                     "Linear Algebra", "Find the rank of a matrix."),
        ]

        summary = TopicAnalysisAgent().summarize(questions)
        plan = StudyPlannerAgent().build_plan(questions, weeks=4, hours_per_week=8, target_score=800)

        self.assertEqual(summary[0]["weighted_marks"], 4)
        self.assertIn("eigenvalues", summary[0]["focus_areas"][2])
        self.assertEqual(plan["paper_years"], [2024, 2025])
        self.assertEqual(len(plan["weeks"]), 4)
        self.assertIn("cannot predict", plan["evidence_note"])

    def test_solution_store_persists_answers(self) -> None:
        with TemporaryDirectory() as directory:
            store = SolutionStore(f"{directory}/solutions.sqlite3")
            store.save("q-1", "Worked solution")

            self.assertEqual(store.load(), {"q-1": "Worked solution"})


if __name__ == "__main__":
    unittest.main()