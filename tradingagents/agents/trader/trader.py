"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent responsible for translating a structured investment plan "
                    "into a concrete transaction proposal. "
                    "Your primary anchor is the Research Manager's investment plan provided below — "
                    "you do not have direct access to the underlying analyst reports. "
                    "Do not invent specific entry prices, stop-loss levels, or price targets unless "
                    "they are explicitly stated in the investment plan. "
                    "If precise price levels are absent from the plan, omit them or mark them as "
                    "'unspecified' in your proposal."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the Research Manager's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"Proposed Investment Plan: {investment_plan}\n\n"
                    f"Based on this plan, produce a specific transaction proposal."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
