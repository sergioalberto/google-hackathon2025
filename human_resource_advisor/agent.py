from google.adk.agents import LlmAgent
from .sub_agents.cv_matcher.agent import cv_matcher_agent
from .sub_agents.cv_searcher.agent import cv_search_agent


RAD_NAME_ID = "projects/570351480416/locations/us-central1/ragCorpora/2738188573441261568"


root_agent = LlmAgent(
        name="cv_master_agent",
        model="gemini-2.0-flash",
        description="Coordinates tasks between curriculum vitaes best matcher and searches",
        instruction=(
            "You are a master coordinator agent. Your goal is to answer user queries that may require combining information from different experts. "
            "You have a CVMatcherAgent and a CVSearchAgent available as sub-agents. "
            "If a question is about finding the best resume for a given job profile, first use the CVMatcherAgent to find the perfect CV match, "
            "then use the CVSearchAgent to answer general questions about the storage CVs. "
            "Clearly state the information found by each expert."
        ),
        sub_agents=[cv_matcher_agent(RAD_NAME_ID), cv_search_agent(RAD_NAME_ID)]
    )
