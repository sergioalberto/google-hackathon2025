# Google Hackathon 2025
A perfect resume for a given job profile using Google ADK


## Setup
```
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux
source .venv/bin/activate

pip install -r requirements.txt

mv .env.example .env
# Add your variables values
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

## Test it
- Upload files from eval/data folder (except `New Job Profile.pdf` file)
- Click on `Index Uploaded CVs` button
- Wait for a couple seconds, until the chat input is enabled
- Ask questions like:
    * `Which is the best candidate for this new job profile?` (and copy&paste the content from `New Job Profile.pdf` file)
    * `Who has more work experience?`
    * Etc

## Build & Deploy
```
gcloud builds submit --tag us-east1-docker.pkg.dev/project-1-test-ai/hackathon/cvs-agents

gcloud run deploy cvs-agents --image us-east1-docker.pkg.dev/project-1-test-ai/hackathon/cvs-agents --platform managed --region us-central1 --allow-unauthenticated
```

### References
- https://google.github.io/adk-docs/
- https://github.com/GoogleCloudPlatform/vertex-ai-creative-studio/blob/main/experiments/mcp-genmedia/sample-agents/adk/README.md
- https://github.com/adhikasp/mcp-linkedin/tree/master
- https://github.com/alinaqi/mcp-linkedin-server
