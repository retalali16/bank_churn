# Bank Churn — Streamlit Deployment

## Folder contents
```
├── streamlit_app.py          # the app (standalone, does not need the notebook)
├── requirements.txt          # Python dependencies
├── data/bank-1.csv           # dataset (already included)
├── .streamlit/secrets.toml   # PLACEHOLDER — put your real Groq key here (local use only)
├── .gitignore                # keeps secrets.toml out of git
└── README.md                 # this file
```

## 1. Run locally
```bash
cd this-folder
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens at http://localhost:8501

To enable the real LLM locally: open `.streamlit/secrets.toml` and replace
`PASTE_YOUR_NEW_GROQ_API_KEY_HERE` with your own key from
https://console.groq.com/keys. Without a key, the app still works — the
"Ask a Question" tab falls back to a clearly-labelled rule-based answer.

## 2. Deploy on Streamlit Community Cloud (free)
1. Create a **new, empty** GitHub repo.
2. Push everything in this folder **except** `.streamlit/secrets.toml`
   (`.gitignore` already excludes it — don't force-add it).
3. Go to https://share.streamlit.io -> "New app" -> connect your repo ->
   set the main file to `streamlit_app.py`.
4. In the app's **Settings -> Secrets**, paste:
   ```
   GROQ_API_KEY = "your-real-key-here"
   ```
5. Deploy. The live app will read the key from Streamlit's secrets manager —
   your key never touches git or the public repo.

## ⚠️ Security note
Never paste a real API key into a chat, a public repo, or a shared file.
If a key has ever been shown anywhere outside your own machine or the
platform's secrets manager, revoke it at https://console.groq.com/keys
and generate a new one.
