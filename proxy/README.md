# FIG OpenAI Proxy

Wraps the OpenAI API so teammates can use it without ever seeing the real API key.

## How it works

```
Teammate code  →  proxy (Render)  →  OpenAI
                  [checks team key]
                  [enforces daily token budget]
```

---

## For the Team Lead — One-time Setup

### 1. Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint** → connect your repo
3. Render will detect `render.yaml` automatically
4. In the Render dashboard, set these **environment variables** (never in the repo):
   - `OPENAI_API_KEY` → your real OpenAI key
   - `TEAM_KEY` → pick any password, e.g. `fig-team-2025`
   - `DAILY_TOKEN_BUDGET` → token limit per day (default: `100000`)
5. Deploy — you'll get a URL like `https://fig-openai-proxy.onrender.com`

### 2. Share with teammates

Tell them:
- **Proxy URL**: `https://fig-openai-proxy.onrender.com`
- **Team Key**: `fig-team-2025` (whatever you set)

That's all they need. They never see your real key.

---

## For Teammates — Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/your-org/FIG.git
cd FIG

# 2. Set up your environment
cp .env.example .env
# Edit .env and fill in OPENAI_API_BASE and OPENAI_API_KEY (team key, not real key)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python -m src.app.dashboard  # or however you run it
```

### Using the proxy in code

The proxy is **drop-in compatible** with the OpenAI SDK — just point it at the proxy URL:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),       # team key
    base_url=os.getenv("OPENAI_API_BASE"),     # proxy URL
)

# Use exactly like normal OpenAI
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Check remaining budget

```bash
curl https://fig-openai-proxy.onrender.com/budget \
  -H "Authorization: Bearer fig-team-2025"
```

---

## Rate Limits

| Setting | Value |
|---|---|
| Daily token budget | 100,000 tokens (shared) |
| Resets | Midnight UTC |
| Auth | Shared team key |

Budget is tracked server-side. When exhausted, requests return `429` until reset.

---

## Adjusting the Budget

In Render dashboard → Environment → change `DAILY_TOKEN_BUDGET` → redeploy.