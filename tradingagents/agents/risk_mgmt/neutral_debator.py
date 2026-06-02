from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies. Here is the trader's decision:

{trader_decision}

Resources available:
{instrument_context}
Market Research Report: {market_research_report}
Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
{fundamentals_label.title()}: {fundamentals_report}
Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last conservative argument: {current_conservative_response}

Respond in three steps:
Step 1 — Quote: Identify the single strongest point made by either the aggressive or conservative analyst in the last round and quote it briefly (or note if no opponent arguments exist yet).
Step 2 — Rebut with evidence: Explain where that point is overly optimistic or overly cautious by citing a specific figure or finding from the reports above. Name the report and the metric.
Step 3 — Stance: One sentence asserting your balanced risk conclusion for this round.

Rules: Do not state any metric, price, or statistic that does not appear in the provided reports above. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
