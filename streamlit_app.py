import os
import time
import uuid
import asyncio
import streamlit as st

from dotenv import load_dotenv
from vertexai.preview import rag
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage
from google.genai import types
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from human_resource_advisor.agent import cv_master_agent

# Load environment variables from .env
load_dotenv()

session_service = InMemorySessionService()

APP_NAME = "agents_app"

# --- Google Cloud Client Initialization ---
# These functions will initialize clients using Application Default Credentials.

@st.cache_resource
def get_gcs_client():
    """Cached GCS client."""
    return storage.Client()

@st.cache_resource
def get_data_store_client():
    """Cached DataStoreServiceClient."""
    return discoveryengine.DataStoreServiceClient()

@st.cache_resource
def get_document_client():
    """Cached DocumentServiceClient."""
    return discoveryengine.DocumentServiceClient()

@st.cache_resource
def get_engine_client():
    """Cached EngineServiceClient."""
    return discoveryengine.EngineServiceClient()

@st.cache_resource
def get_conversational_search_client():
    """Cached ConversationalSearchServiceClient."""
    return discoveryengine.ConversationalSearchServiceClient()

# --- Helper Functions ---

def upload_to_gcs(bucket_name, file_obj, destination_blob_name):
    """Uploads a file object to GCS and returns its GCS URI."""
    try:
        storage_client = get_gcs_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Rewind the file object before upload if it has been read
        file_obj.seek(0)
        blob.upload_from_file(file_obj)
        gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
        # st.info(f"Uploaded {destination_blob_name} to {gcs_uri}")
        return gcs_uri
    except Exception as e:
        st.error(f"Error uploading {destination_blob_name} to GCS: {e}")
        raise


def create_rag(rag_name, model="publishers/google/models/text-embedding-005"):
    # RagCorpus is the resource that contains the RagFiles.
    # Configure embedding model, for example "text-embedding-005".
    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model=model
        )
    )

    rag_corpus = rag.create_corpus(
        display_name=rag_name,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=embedding_model_config
        ),
    )

    time.sleep(15)
    print(f"RagCorpus created: {rag_corpus.name}")
    return rag_corpus


def get_rag(rag_name_id):
    return rag.get_corpus(rag_name_id)


def get_rags():
    return list(rag.list_corpora())


def delete_rag(rag_name_id):
    rag.delete_corpus(rag_name_id)


def import_gcs_files(rag_name: str, gcs_uris: list[str], timeout_secs: int = 600):
    print(f"\nImporting files from {gcs_uris} into RagCorpus...")
    # This starts a long-running operation.
    # The import_files API ingests and indexes the files.
    # We are using a GCS folder URI. It will find all files within.
    import_op = rag.import_files(
        rag_name,
        gcs_uris,  # List of GCS URIs (can be folders or specific files)
        chunk_size=1024,  # Chunks of 1024 tokens
        chunk_overlap=200,  # With 200 tokens overlapping between chunks
        timeout=timeout_secs
    )
    print(f"File import operation initiated. Response/LRO details: {import_op}")
    print(f"Imported files: {import_op.imported_rag_files_count}")
    print(f"Failed files: {import_op.failed_rag_files_count}")

    return import_op


def get_rag_files(rag_name):
    return list(rag.list_files(corpus_name=rag_name))


async def get_agent_session(user_id, session_id):
    if session_id:
        current_session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        if not current_session:
            current_session = await session_service.create_session(app_name=APP_NAME, user_id=user_id)
    else:
        current_session = await session_service.create_session(app_name=APP_NAME, user_id=user_id)
    return current_session


async def talk_with_agents(rag_name: str, user_query: str, conversation_id: str):
    current_session = await get_agent_session(conversation_id, conversation_id)

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=cv_master_agent(rag_name),  # The agent we want to run
        app_name=APP_NAME,   # Associates runs with our app
        session_service=session_service  # Uses our session manager
    )

    print(f"\n>>> User Query: {user_query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=user_query)])

    final_response_text = ""

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=conversation_id, session_id=current_session.id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:  # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            # await runner.close_session(current_session)
            break

    return final_response_text


# Initialize session state variables
is_thinking = False
is_indexing = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "rag_corpus_name" not in st.session_state: # Store the full rag name
    st.session_state.rag_corpus_name = os.environ.get("RAG_CORPUS", "")


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Resumes")
st.title("💬 Chat with your CVs")

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME", "")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("📄 Upload and Index Resumes (PDFs)")

    uploaded_files = st.file_uploader(
        "Upload your PDF documents (recommended that the file name be the same as the CV candidate name)",
        key="uploaded_files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to be indexed."
    )

    if st.button("🚀 Upload files", disabled=not uploaded_files or not project_id or not gcs_bucket_name or is_indexing):
        if uploaded_files and len(uploaded_files) <= 25 and project_id and gcs_bucket_name:
            with st.spinner("Starting indexing process... This may take a significant amount of time. Please be patient."):
                is_indexing = True
                st.session_state.messages = [] # Reset chat on new indexing
                st.session_state.conversation_id = str(uuid.uuid4())

                all_gcs_uris = []
                try:
                    # 1. Upload files to GCS
                    st.info(f"Uploading {len(uploaded_files)} files to GCS bucket: {gcs_bucket_name} ...")
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Sanitize file name for GCS
                        safe_file_name = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in uploaded_file.name)
                        destination_blob_name = f"rag_uploads/{safe_file_name}"
                        gcs_uri = upload_to_gcs(gcs_bucket_name, uploaded_file, destination_blob_name)
                        all_gcs_uris.append(gcs_uri)
                    if not all_gcs_uris:
                        st.error("No files were successfully uploaded to GCS. Aborting.")
                        st.stop()
                    st.success(f"Successfully uploaded {len(all_gcs_uris)} files to GCS.")
                    # st.expander("See GCS URIs").json(all_gcs_uris)

                    # 2. Create RAG (if it does not exist)
                    if not st.session_state.rag_corpus_name:
                        st.info(f"Setting up RAG ...")
                        my_rag = create_rag("RAG for CVs")
                        if my_rag:
                            st.session_state.rag_corpus_name = my_rag.name
                            st.success(f"RAG {my_rag.name} created.")

                    # 3. Import Documents
                    if st.session_state.rag_corpus_name:
                        st.info(f"Importing documents into RAG '{st.session_state.rag_corpus_name}' ...")
                        result = import_gcs_files(st.session_state.rag_corpus_name, all_gcs_uris)
                        if result and result.failed_rag_files_count == 0:
                            st.balloons()
                            st.success("🎉 All setup and indexing steps initiated! You can now try chatting.")
                    is_indexing = False
                except Exception as e:
                    st.error(f"An error occurred during the indexing process: {e}")
                    st.exception(e) # Print full traceback for debugging
                    st.session_state.rag_corpus_name = None # Mark as not ready
                    is_indexing = False
        else:
            st.warning("Please ensure you have uploaded maximum 25 files.")


    if st.button("🧹 Clean", disabled=not st.session_state.rag_corpus_name or is_thinking):
        with st.spinner("Cleaning ..."):
            if st.session_state.rag_corpus_name:
                delete_rag(st.session_state.rag_corpus_name)
                st.session_state.rag_corpus_name = None
                # TODO - Clean up GCS files as well


# --- Chat Interface ---
st.markdown("---")

if not st.session_state.get("rag_corpus_name"):
    st.info("👈 Please upload and successfully index your PDF files first to enable the chat.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your resumes...", disabled=is_thinking):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            try:
                with st.spinner("Thinking..."):
                    is_thinking = True
                    full_response_text = asyncio.run(talk_with_agents(st.session_state.rag_corpus_name,
                                                                      prompt,
                                                                      st.session_state.conversation_id))
                    is_thinking = False

                # The API response includes the updated conversation resource name,
                # which might be useful if the conversation ID changes (though typically it doesn't for ongoing sessions).
                # For this app, we keep st.session_state.conversation_id fixed per "Index PDFs" run.
            except Exception as e:
                full_response_text = f"Sorry, I encountered an error: {e}"
                st.exception(e)

            message_placeholder.markdown(full_response_text)
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates RAG using Vertex AI Search with the ADK.")
