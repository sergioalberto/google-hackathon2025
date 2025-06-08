# Deploying Models

## Deploying Gemma Models to Google Cloud Run
```
gcloud run deploy gemma3-1b \
 --image us-docker.pkg.dev/cloudrun/container/gemma/gemma3-1b \
 --concurrency 2 \
 --cpu 4 \
 --set-env-vars OLLAMA_NUM_PARALLEL=2 \
 --set-env-vars=API_KEY={YOUR_API_KEY} \
 --gpu 1 \
 --gpu-type nvidia-l4 \
 --max-instances 1 \
 --memory 16Gi \
 --allow-unauthenticated \
 --no-cpu-throttling \
 --timeout=600 \
 --region us-central1
```

Note: `YOUR_API_KEY`: Crucial for authentication, set this to a strong, unique API key string of your choice. This key will be required to access your service.

Or
```
gcloud run services replace gemma3-1b.yaml
```


### Testing it
```
curl "<cloud_run_url>/v1beta/models/<model>:generateContent?key={YOUR_API_KEY}" \
   -H 'Content-Type: application/json' \
   -X POST \
   -d '{
     "contents": [{
       "parts":[{"text": "What is a LLM?"}]
       }]
      }'
```

### References
- https://github.com/google-gemini/gemma-cookbook/tree/main/Demos/Gemma-on-Cloudrun
