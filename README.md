# Google Hackathon 2025
A perfect resume for a given job profile using Google ADK


## Setup
```
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run
```
# Web chat app
streamlit run streamlit_app.py

# Dev UI
adk web

# Terminal
adk run

# API server
adk api_server
```

## Build & Deploy
```
gcloud builds submit --tag us-east1-docker.pkg.dev/project-1-test-ai/hackathon/cvs-agents

gcloud run deploy cvs-agents --image us-east1-docker.pkg.dev/project-1-test-ai/hackathon/cvs-agents --platform managed --region us-central1 --allow-unauthenticated
```
