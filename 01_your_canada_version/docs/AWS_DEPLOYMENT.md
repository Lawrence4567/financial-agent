# AWS Deployment Guide

This guide is written for **this repo exactly as it is now**.

## 1. What this app is

Your current app is a single `Streamlit` web app [网页应用] with:

- one entry file: `01_your_canada_version/app/app_local.py`
- local CSV and JSON data [本地数据] inside `01_your_canada_version/data/`
- an optional `OpenAI API` call [接口调用]
- an optional `Yahoo Finance` market-data request [市场数据请求]
- no separate database [数据库] yet
- no separate backend service [后端服务] yet

That means your app is best treated as **one web service [单个 Web 服务]**.

## 2. Best AWS path for this repo

### Recommended now

Use **AWS App Runner** if your AWS account can still create it.

Why it fits:

- very good for one web app
- you do not need to manage servers [服务器]
- you only need a build command, a start command, and environment variables [环境变量]

### Important date

AWS says **App Runner closes to new customers on April 30, 2026**.

That means:

- if your AWS account still has access before that date, App Runner is the fastest path
- if your account does not have access, use the container path with `Dockerfile` plus ECS/Fargate or Lightsail Containers

### Future-proof path

I also added a `Dockerfile` at the repo root. That gives you a container [容器] version of the app, which is useful later for:

- Amazon ECS / Fargate
- Amazon Lightsail Containers
- other container platforms [容器平台]

## 3. Environment variables for this app

These are the variables [变量] your app uses:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `LLM_BACKEND`
- `FINANCE_DATA_DIR`

Recommended values:

- `OPENAI_API_KEY`: your real key
- `OPENAI_MODEL`: `gpt-4o-mini`
- `OPENAI_EMBEDDING_MODEL`: `text-embedding-3-small`
- `LLM_BACKEND`: `langchain`
- `FINANCE_DATA_DIR`: `../data/artifacts_canada`

Notes:

- `OPENAI_API_KEY` is optional. If you leave it empty, the app still runs in rules-only mode [仅规则模式].
- `FINANCE_DATA_DIR` is optional in many cases because your code already has a default path [默认路径].

## 4. Fastest AWS deployment: App Runner from source code

### Step 1. Push your repo to GitHub

App Runner can deploy directly from a GitHub repository [代码仓库].

Make sure these files are pushed:

- `01_your_canada_version/app/app_local.py`
- `01_your_canada_version/requirements-local.txt`
- `01_your_canada_version/data/...`

## Step 2. Open AWS App Runner

In the AWS Console:

1. Search for `App Runner`
2. Click `Create service`
3. Choose `Source code repository`

## Step 3. Connect GitHub

In App Runner:

1. Authorize GitHub if AWS asks
2. Choose your repository
3. Choose the branch you want to deploy

## Step 4. Set the source directory

This repo is not a flat single-app repo. Your app lives in a subfolder [子文件夹].

Set:

- `Source directory`: `01_your_canada_version`

This is important because App Runner runs build and start commands from that folder.

## Step 5. Configure build settings

Use these values:

- `Runtime`: `Python 3`
- `Build command`: `pip install -r requirements-local.txt`
- `Start command`: `streamlit run app/app_local.py --server.address 0.0.0.0 --server.port 8080 --browser.gatherUsageStats false`
- `Port`: `8080`

Why `8080`?

- App Runner asks for a listening port [监听端口]
- `Streamlit` normally uses `8501`, but on App Runner it is easier to make the app listen on the port you configure

## Step 6. Add environment variables

In the App Runner environment variable section, add:

- `OPENAI_API_KEY` = your real API key
- `OPENAI_MODEL` = `gpt-4o-mini`
- `FINANCE_DATA_DIR` = `../data/artifacts_canada`

If you want to test the app without OpenAI first, you can skip `OPENAI_API_KEY`.

## Step 7. Choose service size

Start small [从小开始] first.

Because this app loads local data, creates charts, and may call OpenAI, a very small instance can feel slow. If the UI is sluggish [卡顿], increase memory [内存] first.

## Step 8. Deploy

Click `Create & deploy`.

AWS will:

- pull your code
- install dependencies [依赖]
- run the start command
- give you a public URL [公开网址]

## Step 9. Test the app

After deployment finishes:

1. Open the App Runner URL
2. Check whether the `Copilot`, `Scenario Planner`, and `Dashboard` pages all load
3. Try one rules-only question first
4. Then test one OpenAI question if you added the API key

Good first test questions:

- `What are my monthly spending patterns in Canada?`
- `Based on my profile, should I focus on FHSA, TFSA, or RRSP?`
- `Show me a simple ETF market snapshot.`

## Step 10. If deployment fails

Check App Runner logs [日志] first.

The most likely beginner issues are:

- wrong source directory
- wrong start command
- missing `OPENAI_API_KEY`
- dependency install failure
- app listening on the wrong port

For this repo, the first thing I would re-check is:

- `Source directory` must be `01_your_canada_version`

## 5. Local container test with the Dockerfile I added

Even if you deploy with App Runner first, you should understand the container path [容器路径].

From the repo root:

```powershell
docker build -t financial-advisory-genai .
```

Run it locally:

```powershell
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=your_key_here -e OPENAI_MODEL=gpt-4o-mini financial-advisory-genai
```

Then open:

- `http://localhost:8501`

Why this matters:

- if App Runner is unavailable in your account after April 30, 2026
- if you later move to ECS or Lightsail
- if you want stable local-to-cloud parity [本地与云环境一致]

## 6. How your current code maps to AWS

This is the mental model [思维模型]:

- `Streamlit app` = the web service
- `local CSV/JSON files` = files bundled inside the deployed app image
- `OPENAI_API_KEY` = secret [机密]
- `Yahoo Finance` calls = outbound internet access [对外网络访问]
- `reference_rag_index.json` = prebuilt artifact [预生成产物], so AWS does not need a separate vector database [向量数据库] yet

This is why your app is still relatively easy to deploy.

## 7. What I recommend you do next

Follow this order:

1. If App Runner is still available in your AWS account, deploy there first
2. If not, use the `Dockerfile` path and deploy to ECS/Fargate or Lightsail Containers
3. Verify the public URL works
4. Test rules-only mode
5. Add `OPENAI_API_KEY`
6. Test one LLM question

## 8. Official AWS references

- App Runner getting started: https://docs.aws.amazon.com/apprunner/latest/dg/getting-started.html
- App Runner source code services: https://docs.aws.amazon.com/apprunner/latest/dg/service-source-code.html
- App Runner Python runtime: https://docs.aws.amazon.com/apprunner/latest/dg/service-source-code-python.html
- App Runner availability change: https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html
- Lightsail container services overview: https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-container-services.html
