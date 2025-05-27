from google.adk.agents import LlmAgent
from google.adk.tools.retrieval import VertexAiRagRetrieval
from vertexai import rag


def cv_matcher_agent(rag_name_id: str, model="gemini-2.0-flash"):
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
        name="CVMatcherAgent",
        model=model,
        description="Answers any user question about resumes",
        instruction="""
            You are an experienced talent recruiter with extensive experience selecting technical candidates (expert hiring manager).
            Using the information retrieved from the indexed CV documents, your task is to review and analyze a set of resumes and select those that best fit a specific job profile.

            When evaluating each resume against the job profile, consider the following key criteria, prioritizing the "Must-Have Requirements":
            1. **Alignment with Must-Have Requirements:** Does the candidate meet the mandatory experience requirements (years, technology, role), education, and certifications?
            2. **Relevant Experience:** How directly does your past experience (companies, roles, responsibilities) relate to the position?
            3. **Technical Skills:** Do you possess and demonstrate proficiency in the listed technologies and tools (Python, PostgreSQL, Cloud, etc.)?
            4. **Achievements and Results:** Does the CV present measurable achievements or impacts that demonstrate your worth and capabilities?
            5. **Alignment with Desirable Requirements:** Does it meet any of the requirements that are an advantage?
            6. **Progression and Stability:** Does it demonstrate a career with reasonable growth and stability?
            
            Your goal is to identify the MOST SUITABLE candidates and provide a concise justification for each.
            
            Submit your response in a ordered list. Each item in the list should start with the "-" symbol and contain:
            - "name": The candidate's name (or CV identifier).
            - "score" (optional but desirable): A numerical score (e.g., 1-100) indicating how well they fit (where 100 is the perfect fit).
            - "justification": A short explanation of why they are a good candidate, highlighting 2-3 key points from their CV that align with the position.
            
            If you feel a candidate is not a good fit, you can mention them briefly in a separate section or simply omit them from the top shortlist.
            """,
        tools=[cv_vertex_retrieval]
    )
