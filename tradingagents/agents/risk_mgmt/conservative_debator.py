from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

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

        prompt = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evidence supports reducing exposure, prefer Underweight or Sell framing. Here is the trader's decision:

{trader_decision}

Resources available:
{instrument_context}
Market Research Report: {market_research_report}
Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
{fundamentals_label.title()}: {fundamentals_report}
Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last neutral argument: {current_neutral_response}

Respond in three steps:
Step 1 — Quote: Identify the single strongest point made by the aggressive or neutral analyst in the last round and quote it briefly (or note if no opponent arguments exist yet).
Step 2 — Rebut with evidence: Explain where that point overlooks downside risk by citing a specific figure or finding from the reports above. Name the report and the metric.
Step 3 — Stance: One sentence asserting your conservative risk conclusion for this round.

Rules: Do not state any metric, price, or statistic that does not appear in the provided reports above. Output conversationally as if you are speaking without any special formatting.""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
