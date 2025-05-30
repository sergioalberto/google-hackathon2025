import vertexai
import streamlit as st

from dotenv import load_dotenv
from vertexai.preview import rag
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage
import os
import time
import uuid

# Load environment variables from .env
load_dotenv()

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
        st.info(f"Uploaded {destination_blob_name} to {gcs_uri}")
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


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Resumes")
st.title("📄 Chat with your CVs")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    project_id = st.text_input("Google Cloud Project ID", value=os.environ.get("GOOGLE_CLOUD_PROJECT", ""), help="Your Google Cloud Project ID.")
    gcs_bucket_name = st.text_input("GCS Bucket Name", value=os.environ.get("GCS_BUCKET_NAME", ""), help="GCS bucket to upload PDFs to (must exist and be accessible).")
    rag_name = st.text_input("RAG Patch Name", value=os.environ.get("RAG_CORPUS", ""), help="If you want to use a specific RAG")

    st.markdown("---")
    st.markdown(
        "**Important Notes:**\n"
        "- Ensure the GCS bucket exists and you have permissions.\n"
        "- The Vertex AI Search and Discovery API must be enabled in your project.\n"
        "- Indexing can take several minutes (or longer for many/large PDFs)."
    )

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "rag_corpus_name" not in st.session_state: # Store the full rag name
    st.session_state.rag_corpus_name = None


# --- Main Page ---
st.header("1. Upload and Index Resumes (PDFs)")
uploaded_files = st.file_uploader(
    "Upload your PDF documents", type="pdf", accept_multiple_files=True,
    help="Upload one or more PDF files to be indexed."
)


if st.button("🚀 Index Uploaded CVs", disabled=not uploaded_files or not project_id or not gcs_bucket_name):
    if uploaded_files and project_id and gcs_bucket_name:
        with st.spinner("Starting indexing process... This may take a significant amount of time. Please be patient."):
            st.session_state.messages = [] # Reset chat on new indexing
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.rag_corpus_name = None # Reset engine path

            all_gcs_uris = []
            try:
                # 1. Upload files to GCS
                st.subheader(f"Step 1: Uploading {len(uploaded_files)} files to GCS bucket: {gcs_bucket_name}")
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
                st.expander("See GCS URIs").json(all_gcs_uris)

                # 2. Create RAG (if it does not exist)
                if not rag_name:
                    st.subheader(f"Step 2: Setting up RAG")
                    my_rag = create_rag("RAG for CVs")
                    if my_rag:
                        rag_name = my_rag.name
                        st.success(f"RAG {rag_name} created.")

                # 3. Import Documents
                if rag_name:
                    st.subheader(f"Step 3: Importing documents into RAG '{rag_name}'")
                    result = import_gcs_files(rag_name, all_gcs_uris)
                    if result and result.failed_rag_files_count == 0:
                        st.balloons()
                        st.success("🎉 All setup and indexing steps initiated! You can now try chatting below.")
                        st.session_state.rag_corpus_name = rag_name

            except Exception as e:
                st.error(f"An error occurred during the indexing process: {e}")
                st.exception(e) # Print full traceback for debugging
                st.session_state.rag_corpus_name = None # Mark as not ready

# --- Chat Interface ---
st.markdown("---")
st.header("💬 Chat with your Resumes")

if not st.session_state.get("rag_corpus_name"):
    st.info("☝️ Please upload and successfully index your PDF files first to enable the chat.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            try:
                pass
                # with st.spinner("Thinking..."):
                #    response = conv_client.converse_conversation(request=request)

                # The API response includes the updated conversation resource name,
                # which might be useful if the conversation ID changes (though typically it doesn't for ongoing sessions).
                # For this app, we keep st.session_state.conversation_id fixed per "Index PDFs" run.
            except Exception as e:
                full_response_text = f"Sorry, I encountered an error: {e}"
                st.exception(e)

            message_placeholder.markdown(full_response_text)
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates RAG using Vertex AI Search.")
