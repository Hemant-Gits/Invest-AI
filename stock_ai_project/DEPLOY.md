# Deploy "Invest AI" on Streamlit Cloud

## One-time setup (5 minutes)

### 1. Create a GitHub repository
1. Go to [github.com/new](https://github.com/new)
2. Repository name: `invest-ai` (or `Invest-AI`)
3. Set visibility to **Public** (required for free Streamlit Cloud)
4. Click **Create repository** (do NOT add README)

### 2. Push this project to GitHub
Open PowerShell in this folder (`stock_ai_project`) and run:

```powershell
git init
git add .
git commit -m "Invest AI: Final Year Project"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/invest-ai.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

### 3. Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with **GitHub**
3. Click **Create app** → **Yup, I have an app**
4. Fill in:
   - **Repository:** `YOUR_GITHUB_USERNAME/invest-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** `invest-ai` → your URL becomes `https://invest-ai.streamlit.app`
5. Click **Deploy!**

### 4. Set the display name to "Invest AI"
After deploy:
1. Open your app dashboard on share.streamlit.io
2. Go to **Settings** → **General**
3. Set **App name** to: **Invest AI**
4. Save

### 5. Add API secrets
1. **Settings** → **Secrets**
2. Paste:

```toml
NEWS_API_KEY = "your_key_from_newsapi.org"
```

3. Save — app will auto-redeploy.

---

## Your live URL

```
https://invest-ai-YOUR_USERNAME.streamlit.app
```

or (if you set custom subdomain):

```
https://invest-ai.streamlit.app
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails (memory) | `requirements.txt` already omits TensorFlow for cloud |
| No news articles | Add `NEWS_API_KEY` in Secrets |
| CPI page error | Ensure `data/All India Consumer Price Index.csv` is committed |
| Import errors | Main file path must be `app.py` (not `stock_ai_project/app.py`) |

---

## Important

Push **`stock_ai_project`** contents as the **repo root** (not the parent `InvestAI` folder).
This means `app.py` and `requirements.txt` sit at the top level of the GitHub repo.
