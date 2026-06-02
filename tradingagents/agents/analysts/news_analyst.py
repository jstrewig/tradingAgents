from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_news,
            get_global_news,
        ]

        seven_days_ago = (
            datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=7)
        ).strftime("%Y-%m-%d")

        system_message = (
            f"You are a news analyst tasked with researching recent news relevant to trading and macroeconomics.\n\n"
            f"**Tool call order:**\n"
            f"1. Call `get_news(query, start_date, end_date)` with `start_date = {seven_days_ago}` and "
            f"`end_date = {current_date}` for {asset_label}-specific news "
            f"(use the ticker or {asset_label} name as the query).\n"
            f"2. Call `get_global_news(curr_date, look_back_days=7, limit=20)` for broader macroeconomic "
            f"and sector-level context.\n\n"
            f"**Grounding rules:**\n"
            f"- Cite the article headline and publication date from tool output for every factual claim.\n"
            f"- Do not assert any news event, earnings release, or market development that does not appear "
            f"in the tool response. If the tool returns no results for a query, state that explicitly rather "
            f"than fabricating coverage.\n\n"
            f"**Report structure:**\n"
            f"1. **{asset_label.title()} News** — material events, earnings, product launches, regulatory "
            f"actions, or analyst actions directly concerning the {asset_label}.\n"
            f"2. **Macroeconomic & Sector Context** — relevant macro headlines, sector trends, or "
            f"geopolitical developments from `get_global_news`.\n\n"
            f"Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points, organized and easy to read."
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
            "news_report": report,
        }

    return news_analyst_node
