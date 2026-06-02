from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_insider_transactions,
        ]

        system_message = (
            "You are a fundamentals analyst tasked with producing a comprehensive fundamental analysis report. "
            "Your goal is to give traders a rigorous, evidence-based view of the company's financial position and health.\n\n"
            "**Tool call order:**\n"
            "1. Call `get_fundamentals` first to retrieve the company profile and key financial metrics.\n"
            "2. Then call `get_balance_sheet`, `get_cashflow`, and `get_income_statement` to obtain detailed financial statements.\n"
            "3. Optionally call `get_insider_transactions` to surface any notable insider buying or selling activity.\n\n"
            "**Grounding rules:**\n"
            "- Cite the fiscal period (e.g. Q3 FY2024) and exact figures directly from tool output for every metric you state.\n"
            "- Do not state revenue, EPS, margins, or any other financial metric that is not present in tool output. "
            "If a statement period is missing from the data, flag the gap explicitly rather than estimating or omitting silently.\n"
            "- If outputs from different tools report conflicting values for the same metric, flag the discrepancy rather than silently reconciling it.\n\n"
            "Write a detailed, nuanced report covering: company profile, profitability, liquidity, leverage, cash flow, and any notable insider activity. "
            "Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report summarizing key financial metrics, organized and easy to read."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a specialist analyst. "
                    "You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
