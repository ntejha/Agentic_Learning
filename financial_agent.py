from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
load_dotenv()


# Web search agent

websearch_agent = Agent(
    name = "Web search agent",
    role = "Search the web for the information",
    model = Groq(id = "llama-3.1-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tools_calls = True,
    markdown = True,
)

## Financial agent 

financial_agent = Agent(
    name = "financial agent",
    model = Groq(id = "llama-3.1-70b-versatile"),
    tools = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
        instructions = ["Use tables to display the data"],
        show_tools_calls=True,
        markdown = True,
)


multi_ai_agent=Agent(
    team=[websearch_agent, financial_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)



multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDIA", stream=True)