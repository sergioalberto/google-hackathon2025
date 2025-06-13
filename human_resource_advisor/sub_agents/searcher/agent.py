from google.adk.agents import LlmAgent
from google.adk.tools import google_search


def searcher_agent(model="gemini-2.0-flash"):
    return LlmAgent(
        name="searcher_agent",
        model=model,
        description="Answers any user question",
        instruction="You are a helpful agent who can answer user questions and search externally in Google and Linkedin.",
        tools=[google_search]
    )
