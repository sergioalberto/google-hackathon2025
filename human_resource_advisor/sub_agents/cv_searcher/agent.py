from google.adk.agents import LlmAgent
from google.adk.tools.retrieval import VertexAiRagRetrieval
from vertexai import rag


def cv_search_agent(rag_name_id: str, model="gemini-2.0-flash"):
    cv_vertex_retrieval = VertexAiRagRetrieval(
        name='retrieve_rag_cv_documentation',
        description=(
            'Use this tool to retrieve CV documentation and reference materials for the question from the RAG corpus'
        ),
        rag_resources=[
            rag.RagResource(
                # please fill in your own rag corpus
                # here is a sample rag coprus for testing purpose
                # e.g. projects/123/locations/us-central1/ragCorpora/456
                rag_corpus=rag_name_id
            )
        ],
        similarity_top_k=10,
        vector_distance_threshold=0.5,
    )

    return LlmAgent(
        name="CVSearchAgent",
        model=model,
        description="Answers any user question about resumes",
        instruction="You are a helpful agent who can answer user questions about indexed curriculum vitaes.",
        tools=[cv_vertex_retrieval]
    )
