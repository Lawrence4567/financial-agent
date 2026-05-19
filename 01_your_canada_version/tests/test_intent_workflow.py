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
from intent_engine import IntentSchema
from local_financial_qa import analyze_financial_data_local, answer_from_rules, build_summary, format_spending_answer_from_intent, load_financial_context
from prompt_builder import build_language_instruction, detect_query_language
from retrieval_backend import LocalIndexRetriever


class IntentWorkflowTests(unittest.TestCase):
    def test_food_spending_intent_uses_include_operator(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("how much i spend on food", use_llm=False, chat_history=[])
        self.assertEqual(intent.domain, "spending")
        self.assertEqual(intent.operator, "include")
        self.assertIn("Food", intent.target_entities)

    def test_food_spending_in_three_months_uses_latest_three_month_window(self) -> None:
        summary = build_summary(load_financial_context())
        answer = answer_from_rules("how much i spend on food in 3 month", summary)
        self.assertIn("latest 3 months", answer)
        self.assertIn("October 2025 to December 2025", answer)
        self.assertIn("$664.70", answer)
        self.assertNotIn("1,189.46", answer)

    def test_food_spending_without_time_window_uses_full_loaded_window(self) -> None:
        summary = build_summary(load_financial_context())
        answer = answer_from_rules("How much did I spend on food?", summary)
        self.assertIn("12-month client record", answer)
        self.assertIn("January 2025 to December 2025", answer)
        self.assertIn("2,544.33", answer)

    def test_chinese_food_spending_uses_gpt_intent_category_and_local_time_scope(self) -> None:
        summary = build_summary(load_financial_context())
        intent = IntentSchema(domain="spending", operator="include", target_entities=["Food"], confidence="high")
        query = "\u6211\u6700\u8fd13\u4e2a\u6708\u98df\u7269\u82b1\u4e86\u591a\u5c11\u94b1"
        answer = format_spending_answer_from_intent(summary, intent, query)
        self.assertIn("latest 3 months", answer)
        self.assertIn("664.70", answer)

    def test_rent_spending_in_three_months_uses_latest_three_month_window(self) -> None:
        summary = build_summary(load_financial_context())
        answer = answer_from_rules("how much did I spend on rent in 3 months", summary)
        self.assertIn("latest 3 months", answer)
        self.assertIn("7,050.00", answer)

    def test_chinese_this_month_spending_routes_to_latest_available_month(self) -> None:
        query = "\u6211\u8fd9\u4e2a\u6708\u82b1\u4e86\u591a\u5c11\u94b1"
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent(query, use_llm=False, chat_history=[])
        plan = plan_capabilities(intent, query)
        summary = build_summary(load_financial_context())
        answer = answer_from_rules(query, summary)
        self.assertEqual(intent.domain, "spending")
        self.assertEqual(plan.tool_calls, ["spending_tool"])
        self.assertIn("latest available month", answer)
        self.assertIn("December 2025 to December 2025", answer)
        self.assertIn("6,209.19", answer)

    def test_general_spending_latest_two_months_uses_time_window(self) -> None:
        summary = build_summary(load_financial_context())
        answer = answer_from_rules("how much i spend this in latest 2 month", summary)
        self.assertIn("latest 2 months", answer)
        self.assertIn("November 2025 to December 2025", answer)
        self.assertIn("11,010.63", answer)
        self.assertNotIn("six-month sample", answer)

    def test_living_spending_latest_month_answers_living_amount_first(self) -> None:
        summary = build_summary(load_financial_context())
        answer = answer_from_rules("How much i spend on living in latest month?", summary)
        self.assertIn("living expenses were `$4,163.59`", answer)
        self.assertIn("Total cash outflows: $6,209.19", answer)
        self.assertIn("Largest living categories: Rent $2,350.00", answer)

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

    def test_negative_recommendation_answer_uses_avoid_wording(self) -> None:
        summary = build_summary(load_financial_context())
        answer = answer_from_rules("What should I avoid recommending right now?", summary)
        self.assertIn("avoid", answer.lower())
        self.assertIn("lower-priority", answer.lower())

    def test_fhsa_vs_tfsa_routes_to_knowledge_compare(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("FHSA vs TFSA", use_llm=False, chat_history=[])
        plan = plan_capabilities(intent, "FHSA vs TFSA")
        self.assertEqual(intent.domain, "knowledge")
        self.assertEqual(intent.operator, "compare")
        self.assertTrue(intent.needs_rag)
        self.assertEqual(plan.tool_calls, ["reference_retrieval"])

    def test_fhsa_tfsa_rrsp_priority_uses_recommendation_and_rag(self) -> None:
        query = "Based on my profile, should I focus on FHSA, TFSA, or RRSP?"
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent(query, use_llm=False, chat_history=[])
        plan = plan_capabilities(intent, query)
        self.assertEqual(intent.domain, "recommendation")
        self.assertTrue(intent.needs_rag)
        self.assertTrue(intent.needs_recommendation)
        self.assertEqual(plan.legacy_route, "hybrid_rule_advice")
        self.assertIn("recommendation_engine", plan.tool_calls)
        self.assertIn("reference_retrieval", plan.tool_calls)

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

    def test_market_explanation_prefers_market_answer_over_performance_draft(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            result = analyze_financial_data_local(
                "Explain market changes and what they mean for this portfolio.",
                use_llm=False,
                request_metadata={
                    "user_id": "501",
                    "session_id": "session-market-answer",
                    "channel": "web_app",
                    "device": "browser",
                    "timestamp": "2026-04-21T12:05:00Z",
                    "sso_provider": "demo_local_login",
                },
            )
        self.assertEqual(result["intent"]["domain"], "market")
        self.assertIn("market", result["answer"].lower())
        self.assertNotIn("The short answer: 2025-06", result["answer"])

    def test_rag_routing_ignores_plain_account_summary_even_if_flagged(self) -> None:
        intent = IntentSchema(
            domain="account",
            operator="summarize",
            needs_rag=True,
            confidence="high",
        )
        plan = plan_capabilities(intent, "Show my household account summary.")
        self.assertIn("account_summary_tool", plan.tool_calls)
        self.assertNotIn("reference_retrieval", plan.tool_calls)

    def test_follow_up_query_keeps_recommendation_context(self) -> None:
        history = [
            {"role": "user", "content": "What should I focus on first?"},
            {"role": "assistant", "content": "Start with FHSA, then TFSA.", "route_label": "Recommendation rules"},
        ]
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            intent = parse_intent("what about that one", use_llm=False, chat_history=history)
        self.assertEqual(intent.domain, "recommendation")

    def test_why_follow_up_after_hybrid_recommendation_explains_reasoning(self) -> None:
        metadata = {
            "user_id": "501",
            "session_id": "session-follow-up-why",
            "channel": "web_app",
            "device": "browser",
            "timestamp": "2026-04-21T12:30:00Z",
            "sso_provider": "demo_local_login",
        }
        first_query = "Based on my profile, should I focus on FHSA, TFSA, or RRSP?"
        with patch.dict(os.environ, {"INTENT_BACKEND": "rules_fallback"}, clear=False):
            first_result = analyze_financial_data_local(first_query, use_llm=False, chat_history=[], request_metadata=metadata)
            history = [
                {"role": "user", "content": first_query},
                {
                    "role": "assistant",
                    "content": first_result["answer"],
                    "route_label": first_result["route_label"],
                    "user_query": first_query,
                    "effective_query": first_result.get("effective_query"),
                },
                {"role": "user", "content": "why?"},
            ]
            follow_up = analyze_financial_data_local("why?", use_llm=False, chat_history=history, request_metadata=metadata)
        self.assertTrue(follow_up["conversation_resolution"]["is_follow_up"])
        self.assertEqual(follow_up["conversation_resolution"]["style"], "why")
        self.assertIn("Why is FHSA", follow_up["effective_query"])
        self.assertIn("comes out well", follow_up["answer"])
        self.assertNotEqual(first_result["answer"], follow_up["answer"])

    def test_language_instruction_prefers_current_english_question(self) -> None:
        query = "What product fits my goals best right now?"
        self.assertEqual(detect_query_language(query), "english")
        self.assertIn("answer in English by default", build_language_instruction(query))

    def test_language_instruction_understands_chinese_but_defaults_to_english(self) -> None:
        query = "FHSA和TFSA我该先选哪个？"
        self.assertEqual(detect_query_language(query), "simplified_chinese")
        self.assertIn("answer in English by default", build_language_instruction(query))

    def test_language_instruction_allows_explicit_chinese_request(self) -> None:
        query = "请用中文回答：FHSA和TFSA我该先选哪个？"
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
        self.assertEqual(result["evidence_validation"]["status"], "ok")

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
