import os
import pandas as pd
from agno.agent import Agent
from agno.models.groq import Groq 
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

from dotenv import load_dotenv
load_dotenv()

web_agent = Agent(
    name = "Web search Agent",
    role = "Search the web information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools = [DuckDuckGoTools()],
    instructions = ["Always use this tool to search the web for information."],
    show_tool_calls= True,
    markdown = True,
)

finance_agent = Agent(
    name = "Finance agent",
    model= Groq(id="llama-3.3-70b-versatile"),
    tools = [
        YFinanceTools(stock_price = True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions = ["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)

multi_agent = Agent(
    name="A stock market Agent",
    role="A comprehensive assistant specializing in stock market analysis by combining financial insights with real-time web searches to deliver accurate, up-to-date information",
    model=Groq(api_key=os.getenv('Groq_API_KEY')),
    team= [web_agent, finance_agent],
    instructions= ["Always use this tool to search the web for information.","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_agent.print_response("Summarize analyst recommendation and share the latest news on HCL.", stream=True)