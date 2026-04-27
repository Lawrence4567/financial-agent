from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from intent_engine import parse_intent, plan_capabilities
from local_financial_qa import analyze_financial_data_local
from prompt_builder import build_language_instruction, detect_query_language
from retrieval_backend import LocalIndexRetriever


class IntentWorkflowTests(unittest.TestCase):
    def test_food_spending_intent_uses_include_operator(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("how much i spend on food", use_llm=False, chat_history=[])
        self.assertEqual(intent.domain, "spending")
        self.assertEqual(intent.operator, "include")
        self.assertIn("Food", intent.target_entities)

    def test_non_food_spending_intent_uses_exclude_operator(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("how much i not spend on food", use_llm=False, chat_history=[])
        self.assertEqual(intent.domain, "spending")
        self.assertEqual(intent.operator, "exclude")
        self.assertEqual(intent.polarity, "negative")

    def test_negative_recommendation_intent_deprioritizes(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("what should i avoid", use_llm=False, chat_history=[])
        plan = plan_capabilities(intent, "what should i avoid")
        self.assertEqual(intent.domain, "recommendation")
        self.assertEqual(intent.operator, "deprioritize")
        self.assertEqual(plan.tool_calls, ["recommendation_engine"])

    def test_fhsa_vs_tfsa_routes_to_knowledge_compare(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("FHSA vs TFSA", use_llm=False, chat_history=[])
        plan = plan_capabilities(intent, "FHSA vs TFSA")
        self.assertEqual(intent.domain, "knowledge")
        self.assertEqual(intent.operator, "compare")
        self.assertTrue(intent.needs_rag)
        self.assertEqual(plan.tool_calls, ["reference_retrieval"])

    def test_spending_and_rules_query_uses_recommendation_and_rag(self) -> None:
        query = "based on my spending and rules, what should I do?"
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent(query, use_llm=False, chat_history=[])
        plan = plan_capabilities(intent, query)
        self.assertEqual(intent.domain, "recommendation")
        self.assertIn("spending_tool", plan.tool_calls)
        self.assertIn("recommendation_engine", plan.tool_calls)
        self.assertIn("reference_retrieval", plan.tool_calls)

    def test_numeric_finance_domains_use_deterministic_answers(self) -> None:
        examples = {
            "Explain my current portfolio allocation.": "portfolio",
            "Did I make or lose money this month?": "performance",
            "What is happening in the market?": "market",
        }
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            for query, expected_domain in examples.items():
                with self.subTest(query=query):
                    intent = parse_intent(query, use_llm=False, chat_history=[])
                    plan = plan_capabilities(intent, query)
                    self.assertEqual(intent.domain, expected_domain)
                    self.assertFalse(plan.requires_generation)
                    self.assertFalse(plan.prefers_llm)

    def test_follow_up_query_keeps_recommendation_context(self) -> None:
        history = [
            {"role": "user", "content": "What should I focus on first?"},
            {"role": "assistant", "content": "Start with FHSA, then TFSA.", "route_label": "Recommendation rules"},
        ]
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("what about that one", use_llm=False, chat_history=history)
        self.assertEqual(intent.domain, "recommendation")

    def test_language_instruction_prefers_current_english_question(self) -> None:
        query = "What product fits my goals best right now?"
        self.assertEqual(detect_query_language(query), "english")
        self.assertIn("Answer in English.", build_language_instruction(query))

    def test_language_instruction_detects_chinese_with_account_acronyms(self) -> None:
        query = "FHSA和TFSA我该先选哪个？"
        self.assertEqual(detect_query_language(query), "simplified_chinese")
        self.assertIn("Answer in Simplified Chinese.", build_language_instruction(query))

    def test_local_index_retriever_contract(self) -> None:
        retriever = LocalIndexRetriever()
        result = retriever.retrieve("FHSA vs TFSA", top_k=2, filters={"topic": "registered_accounts"})
        self.assertEqual(result.backend, "local_index")
        self.assertEqual(result.query_used, "FHSA vs TFSA")
        self.assertEqual(result.filters_applied["topic"], "registered_accounts")
        self.assertLessEqual(len(result.chunks), 2)

    def test_analysis_result_exposes_intent_and_capability_plan(self) -> None:
        result = analyze_financial_data_local(
            "how much i not spend on food?",
            use_llm=False,
            request_metadata={
                "user_id": "501",
                "session_id": "session-intent-test",
                "channel": "web_app",
                "device": "browser",
                "timestamp": "2026-04-21T12:00:00Z",
                "sso_provider": "demo_local_login",
            },
        )
        self.assertEqual(result["intent"]["operator"], "exclude")
        self.assertEqual(result["capability_plan"]["tool_calls"], ["spending_tool"])
        self.assertEqual(result["generation_source"], "rules_fallback")
        self.assertIn("outside `Food`", result["answer"])

    def test_spending_answer_exposes_local_dataset_citation(self) -> None:
        result = analyze_financial_data_local(
            "how much i spend on food?",
            use_llm=False,
            request_metadata={
                "user_id": "501",
                "session_id": "session-citation-spending",
                "channel": "web_app",
                "device": "browser",
                "timestamp": "2026-04-21T12:10:00Z",
                "sso_provider": "demo_local_login",
            },
        )
        labels = [item["label"] for item in result["answer_citations"]]
        self.assertIn("Transaction history", labels)

    def test_knowledge_answer_exposes_retrieved_reference_citation(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            result = analyze_financial_data_local(
                "FHSA vs TFSA",
                use_llm=False,
                request_metadata={
                    "user_id": "501",
                    "session_id": "session-citation-rag",
                    "channel": "web_app",
                    "device": "browser",
                    "timestamp": "2026-04-21T12:20:00Z",
                    "sso_provider": "demo_local_login",
                },
            )
        kinds = [item["kind"] for item in result["answer_citations"]]
        self.assertIn("retrieved_reference", kinds)

    def test_recommendation_answer_and_cards_stay_aligned_to_top_three(self) -> None:
        result = analyze_financial_data_local(
            "What product fits my goals best?",
            use_llm=False,
            request_metadata={
                "user_id": "501",
                "session_id": "session-recommendation-alignment",
                "channel": "web_app",
                "device": "browser",
                "timestamp": "2026-04-21T12:30:00Z",
                "sso_provider": "demo_local_login",
            },
        )
        self.assertEqual(result["route_label"], "Recommendation rules")
        self.assertLessEqual(len(result["recommendation_cards"]), 3)
        self.assertEqual(
            len(result["tool_outputs"]["recommendation_engine"]["scored_recommendations"]),
            len(result["recommendation_cards"]),
        )
        self.assertNotIn("The recommendation cards below show", result["answer"])


if __name__ == "__main__":
    unittest.main()
