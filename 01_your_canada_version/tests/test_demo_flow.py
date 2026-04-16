from __future__ import annotations

import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from local_financial_qa import analyze_financial_data_local
from query_router import route_query


class DemoFlowTests(unittest.TestCase):
    def test_portfolio_down_query_routes_to_performance_explanation(self) -> None:
        decision = route_query("Why did my portfolio go down this month?", use_llm=False)
        self.assertEqual(decision.route, "performance_explanation")
        self.assertTrue(decision.uses_rag)

    def test_performance_demo_answer_includes_exact_month_clarification(self) -> None:
        result = analyze_financial_data_local(
            "Why did my portfolio go down this month?",
            use_llm=False,
            request_metadata={
                "user_id": "501",
                "session_id": "session-test",
                "channel": "web_app",
                "device": "browser",
                "timestamp": "2026-04-15T12:00:00Z",
                "sso_provider": "demo_local_login",
            },
        )
        self.assertEqual(result["route_label"], "Performance explanation")
        self.assertIn("2025-06", result["answer"])
        self.assertIn("2025-04", result["answer"])
        self.assertEqual(result["tool_outputs"]["tool_name"], "portfolio_performance_toolkit")
        self.assertTrue(result["access_decision"]["allowed"])

    def test_access_is_denied_for_wrong_user(self) -> None:
        result = analyze_financial_data_local(
            "Show my account summary",
            use_llm=False,
            request_metadata={
                "user_id": "999",
                "session_id": "session-test-denied",
                "channel": "mobile_app",
                "device": "ios_phone",
                "timestamp": "2026-04-15T12:00:00Z",
                "sso_provider": "demo_local_login",
            },
        )
        self.assertEqual(result["mode"], "access_denied")
        self.assertFalse(result["access_decision"]["allowed"])

    def test_market_context_question_routes_to_market_explanation(self) -> None:
        decision = route_query(
            "Explain market changes and what they mean for this portfolio.",
            use_llm=False,
        )
        self.assertEqual(decision.route, "market_explanation")


if __name__ == "__main__":
    unittest.main()
