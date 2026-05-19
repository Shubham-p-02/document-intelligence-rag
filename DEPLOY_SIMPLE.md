# Put your app on the internet (simple guide)

You only need a **free Hugging Face account**, a **free Groq API key**, and about **15 minutes**.  
Do **not** paste your API key into GitHub—only into Hugging Face **Secrets** (Step 4).

---

## Step 1 — Open the “new Space” page

**Click this link:** https://huggingface.co/new-space

**What you should see:** A form titled something like “Create a new Space.” If it asks you to **log in** or **sign up**, do that first, then open the link again.

---

## Step 2 — Fill in the form (copy these choices)

| Field | What to choose |
|--------|----------------|
| **Owner** | Your Hugging Face username (for example `shubhamp02`) |
| **Space name** | `document-intelligence-rag` |
| **License** | Any (e.g. MIT) — required by the site |
| **Select the Space SDK** | **Streamlit** |
| **Space hardware** | **CPU basic** (free) |
| **Visibility** | **Public** |

Then click **Create Space**.

**What you should see:** A new page for your Space. It may say **Building** or show a log— that is normal.

---

## Step 3 — Put the project code on the Space

You need the same files as this GitHub repo:  
https://github.com/Shubham-p-02/document-intelligence-rag

Pick **one** of these ways (easiest first):

### Way A — Connect GitHub (best if you see the option)

1. On your Space page, open **Settings** (gear icon or “Settings” tab).
2. Look for **Repository** or **GitHub** / **Sync with a git repository**.
3. Connect your GitHub account if asked, then choose repository **`Shubham-p-02/document-intelligence-rag`**, branch **`main`**.
4. Save. The Space will pull the code automatically.

**What you should see:** Files like `app.py`, `streamlit_app.py`, and `requirements.txt` listed under the Space **Files** tab after a short wait.

### Way B — Duplicate on Hugging Face (no GitHub linking)

1. Open https://huggingface.co/new-space again only if you need a fresh Space—or stay on your Space.
2. On https://huggingface.co/spaces — find any **Streamlit** demo you like, or use Hugging Face’s **“Duplicate this Space”** from a template, **or** upload files manually:
3. On your Space, go to **Files** → **Add file** → **Upload files** (you may need several uploads), and upload at least from your computer’s copy of the project folder:
   - `streamlit_app.py`, `app.py`, `requirements.txt`, `.streamlit/config.toml`
   - `ingest.py`, `chunking.py`, `vectorstore.py`, `chain.py`
   - the `sample_corpus` folder files

**Easier Way B shortcut:** On GitHub, click **Code** → **Download ZIP**, unzip, then drag the important files into the Space **Files** upload.

**What you should see:** A **Files** list that includes `streamlit_app.py` and `requirements.txt`. The Space may restart and show **Building** again.

---

## Step 4 — Add your Groq API key (secret)

1. Get a free key: https://console.groq.com/ → sign up → create an API key → copy it (starts with `gsk_`).
2. On your Space, open **Settings** → **Repository secrets** (sometimes labeled **Secrets**).
3. Click **New secret** (or **Add a new secret**).
4. **Name:** `GROQ_API_KEY` (type exactly, all caps with underscores).
5. **Value:** paste your Groq key.
6. Save.

**What you should see:** A secret named `GROQ_API_KEY` in the list. It will **not** show the full key again—that is correct.  
**Never** commit this key to GitHub or upload it as a file.

---

## Step 5 — Open the live app

1. Click the **App** tab at the top of your Space (not “Logs” only).
2. Wait **2–5 minutes** while status says **Building** (first time can take longer while packages install).
3. When status is **Running**, the Streamlit app should appear in the browser.

**What you should see:**

- A Streamlit page with a title about document / RAG.
- A place to pick a **corpus folder** (use `sample_corpus` if it is in the repo).
- A button like **Build / rebuild index** — click it once and wait.
- Then you can **type a question** and get an answer.

**Your public link to share:**

`https://huggingface.co/spaces/YOUR_HF_USERNAME/document-intelligence-rag`

Replace `YOUR_HF_USERNAME` with your real Hugging Face username (example: `https://huggingface.co/spaces/shubhamp02/document-intelligence-rag`).

---

## If something goes wrong

| Problem | What to try |
|--------|-------------|
| **Building forever** | Open **Logs** tab; look for red errors. Often missing `requirements.txt` or wrong SDK (must be **Streamlit**). |
| **App says missing API key** | Repeat Step 4; secret name must be exactly `GROQ_API_KEY`. |
| **Blank page** | Refresh; wait 5 more minutes on free CPU. |
| **Cannot find GitHub repo** | Use Way B (upload / ZIP) in Step 3. |

---

## Done?

Share your Space **App** URL with recruiters. You do **not** need to redeploy GitHub—only the Hugging Face Space hosts the live demo.
