# AWS Deployment Guide

Use this document only for cloud deployment.

For project overview, local setup, and architecture links, start from the root `README.md`.

## 1. Best AWS path for this repo

### Recommended now

Use **AWS App Runner** if your AWS account can still create it.

Why it fits:

- the repo deploys as one web service
- you do not need to manage servers
- you mainly need a build command, a start command, and environment variables

### Important date

AWS says **App Runner closes to new customers on April 30, 2026**.

That means:

- if your AWS account still has access before that date, App Runner is the fastest path
- if your account does not have access, use the `Dockerfile` path with ECS/Fargate or Lightsail Containers

### Future-proof path

The repo also includes a `Dockerfile` at the root. That gives you a container version of the app, which is useful later for:

- Amazon ECS / Fargate
- Amazon Lightsail Containers
- other container platforms

## 2. Environment variables for this app

These are the main variables the app uses:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `LLM_BACKEND`
- `INTENT_BACKEND`
- `RETRIEVAL_BACKEND`
- `WORKFLOW_BACKEND`
- `FINANCE_DATA_DIR`

Recommended values:

- `OPENAI_API_KEY`: your real key
- `OPENAI_MODEL`: `gpt-5.4`
- `OPENAI_EMBEDDING_MODEL`: `text-embedding-3-small`
- `LLM_BACKEND`: `langchain`
- `INTENT_BACKEND`: `llm`
- `RETRIEVAL_BACKEND`: `local_index`
- `WORKFLOW_BACKEND`: `langgraph`
- `FINANCE_DATA_DIR`: `../data/artifacts_canada`

Notes:

- `OPENAI_API_KEY` is optional. If you leave it empty, the app still runs with deterministic fallback answers, but LLM intent parsing and LLM synthesis are disabled.
- `RETRIEVAL_BACKEND=local_index` is the default and does not require any external vector database.
- `FINANCE_DATA_DIR` is optional in many cases because the code already has a default path.

## 3. Fastest AWS deployment: App Runner from source code

### Step 1. Push your repo to GitHub

App Runner can deploy directly from a GitHub repository.

Make sure these files are pushed:

- `01_your_canada_version/app/app_local.py`
- `01_your_canada_version/requirements-local.txt`
- `01_your_canada_version/data/...`

### Step 2. Open AWS App Runner

In the AWS Console:

1. Search for `App Runner`
2. Click `Create service`
3. Choose `Source code repository`

### Step 3. Connect GitHub

In App Runner:

1. Authorize GitHub if AWS asks
2. Choose your repository
3. Choose the branch you want to deploy

### Step 4. Set the source directory

This repo is not a flat single-app repo. The app lives in a subfolder.

Set:

- `Source directory`: `01_your_canada_version`

This is important because App Runner runs build and start commands from that folder.

### Step 5. Configure build settings

Use these values:

- `Runtime`: `Python 3`
- `Build command`: `pip install -r requirements-local.txt`
- `Start command`: `streamlit run app/app_local.py --server.address 0.0.0.0 --server.port 8080 --browser.gatherUsageStats false`
- `Port`: `8080`

Why `8080`?

- App Runner asks for a listening port
- `Streamlit` normally uses `8501`, but on App Runner it is easier to make the app listen on the port you configure

### Step 6. Add environment variables

In the App Runner environment variable section, add:

- `OPENAI_API_KEY` = your real API key
- `OPENAI_MODEL` = `gpt-5.4`
- `INTENT_BACKEND` = `llm`
- `RETRIEVAL_BACKEND` = `local_index`
- `WORKFLOW_BACKEND` = `langgraph`
- `FINANCE_DATA_DIR` = `../data/artifacts_canada`

If you want to test the app without OpenAI first, you can skip `OPENAI_API_KEY`.

### Step 7. Choose service size

Start small first.

Because this app loads local data, creates charts, and may call OpenAI, a very small instance can feel slow. If the UI feels sluggish, increase memory first.

### Step 8. Deploy

Click `Create & deploy`.

AWS will:

- pull your code
- install dependencies
- run the start command
- give you a public URL

### Step 9. Test the app

After deployment finishes:

1. Open the App Runner URL
2. Check whether the `Copilot`, `Scenario Planner`, and `Dashboard` pages all load
3. Try one rules-only question first
4. Then test one OpenAI question if you added the API key

Good first test questions:

- `What are my monthly spending patterns in Canada?`
- `Based on my profile, should I focus on FHSA, TFSA, or RRSP?`
- `Show me a simple ETF market snapshot.`

### Step 10. If deployment fails

Check App Runner logs first.

The most likely beginner issues are:

- wrong source directory
- wrong start command
- missing `OPENAI_API_KEY`
- dependency install failure
- app listening on the wrong port

For this repo, the first thing to re-check is:

- `Source directory` must be `01_your_canada_version`

## 4. Local container test with the Dockerfile

Even if you deploy with App Runner first, you should understand the container path.

From the repo root:

```powershell
docker build -t financial-advisory-genai .
```

Run it locally:

```powershell
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=your_key_here -e OPENAI_MODEL=gpt-5.4 financial-advisory-genai
```

Then open:

- `http://localhost:8501`

Why this matters:

- if App Runner is unavailable in your account after April 30, 2026
- if you later move to ECS or Lightsail
- if you want stable local-to-cloud parity

## 5. How the current code maps to AWS

This is the mental model:

- `Streamlit app` = the web service
- `local CSV/JSON files` = files bundled inside the deployed app image
- `OPENAI_API_KEY` = secret
- `Yahoo Finance` calls = outbound internet access
- `reference_rag_index.json` = the default local retrieval backend artifact, so AWS does not need a separate vector database yet
- `INTENT_BACKEND` + `WORKFLOW_BACKEND` = the knobs that control whether the app uses the new hybrid GenAI workflow end to end

This is why the app is still relatively easy to deploy.

## 6. What I recommend you do next

Follow this order:

1. If App Runner is still available in your AWS account, deploy there first
2. If not, use the `Dockerfile` path and deploy to ECS/Fargate or Lightsail Containers
3. Verify the public URL works
4. Test rules-only mode
5. Add `OPENAI_API_KEY`
6. Test one LLM question

## 7. Official AWS references

- App Runner getting started: https://docs.aws.amazon.com/apprunner/latest/dg/getting-started.html
- App Runner source code services: https://docs.aws.amazon.com/apprunner/latest/dg/service-source-code.html
- App Runner Python runtime: https://docs.aws.amazon.com/apprunner/latest/dg/service-source-code-python.html
- App Runner availability change: https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html
- Lightsail container services overview: https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-container-services.html
