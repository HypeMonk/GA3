# GA3 Universal Guide — START HERE (Index)

A complete, step-by-step guide to all **13 questions** of `exam-tds-2026-05-ga3`.
Every question is personalised **only by your email**, so the same method works for
everyone — you just feed in *your* downloaded data.

The guide is split into 2 main files so nothing is overwhelming:

| File | Covers | What you do |
| :--- | :--- | :--- |
| **this file** | Overview, setup, submission map | Read once |
| **`PART-A.md`** | Q1, Q5, Q10, Q11, Q12, Q13 | Run a small script, paste the answer |
| **`PART-B.md`** | Q2, Q3, Q4, Q6, Q7, Q8, Q9 | Deploy **one** app, submit its URL 7× |


---

## 0. One-time setup

### 0.1 Install tools
```bash
# Python 3.11+ and uv (fast Python runner)
pip install uv

# yt-dlp (for Q1) and ffmpeg (for Q6 audio)
pip install yt-dlp
# ffmpeg: Windows -> https://ffmpeg.org/download.html ; Mac -> brew install ffmpeg
```

### 0.2 Get your AIPipe token (your college key)
1. Open **https://aipipe.org/login** and sign in with your IITM Google account.
2. After login you land on a page showing your **token** — copy it.
3. Keep it handy; you'll paste it into the scripts and the deploy config as
   `AIPIPE_TOKEN`.

AIPipe is an OpenAI-compatible proxy. Everything uses:
- Base URL: `https://aipipe.org/openai/v1`
- Header: `Authorization: Bearer <YOUR_AIPIPE_TOKEN>`
- Budget: **100 cents / week** — plenty for this assignment.

### 0.3 Hugging Face account
Create a free account at **https://huggingface.co** (needed for Bucket B).

---

## 1. How GA3 works (read this once)

- Each question shows either a **Download** / **Copy** button (your seeded data) or
  an **iframe** with assigned values. **Always use your own** — never a friend's.
- Two ways to answer:
  - **Paste** a JSON / number / file (Bucket A).
  - **Submit a URL** (Bucket B). The grader then POSTs hidden, seeded test data to
    your live endpoint and checks the response.
- For URL questions, **keep your app running** until you click **Check**, see
  **Correct**, and then **Submit**.

---

## 2. Submission map (the important table)

After you deploy the Bucket B app (see `ga3-bucket-b-deploy.md`), your base URL
looks like `https://<user>-<space>.hf.space`. Submit exactly this:

| Q | Bucket | What to submit |
| :--- | :--- | :--- |
| **Q1** | A | Paste JSON `{"urls":[...]}` |
| **Q2** | B | `https://<user>-<space>.hf.space` *(base URL — grader adds `/answer-image`)* |
| **Q3** | B | `https://<user>-<space>.hf.space` *(base URL — grader adds `/extract`)* |
| **Q4** | B | `https://<user>-<space>.hf.space` *(base URL — grader adds `/dynamic-extract`)* |
| **Q5** | A | Paste JSON `{"Q001":[...], ...}` |
| **Q6** | B | `https://<user>-<space>.hf.space/answer-audio` *(full URL)* |
| **Q7** | B | `https://<user>-<space>.hf.space/extract` *(full URL)* |
| **Q8** | B | `https://<user>-<space>.hf.space/rank` *(full URL)* |
| **Q9** | B | `https://<user>-<space>.hf.space/solve` *(full URL)* |
| **Q10** | A | Paste the nonce (a number) |
| **Q11** | A | Paste JSON `{"answers":{...},"token_counts":{...},"pipeline_code":"..."}` |
| **Q12** | A | Paste `session.cast` contents |
| **Q13** | A | Paste JSON `{"q1":"p-017", ...}` |

> **Why Q2/Q3/Q4 use the base URL but Q6–Q9 use the full path:** that's how each
> question's verifier is written in the exam — Q2/Q3/Q4 append the path for you,
> the others call exactly the URL you submit.

---

## 3. What's guaranteed vs best-effort (be realistic)

| Confidence | Questions | Why |
| :--- | :--- | :--- |
| ✅ **Guaranteed** (pure computation / values printed) | Q1, Q5, Q10, Q11, Q12 | Deterministic — no model, no luck |
| ✅ **High** (embeddings / strict schema) | Q8, Q13, Q3, Q4, Q7, Q9 | Robust with the models below |
| 🟡 **Best-effort** | Q2 (chart/receipt reading), Q6 (Korean audio) | Depends on model quality; Q6 is the hardest |

**Realistic expectation:** ~11–13 marks are very achievable. Do the guaranteed ones
first so you can't lose them, then push on the model-dependent ones.

---

## 4. Models used (all via AIPipe)

| Task | Model | Verified on AIPipe |
| :--- | :--- | :--- |
| Text extraction / reasoning (Q3,4,7,9) | `gpt-4o-mini` | ✅ |
| Vision (Q2) | `gpt-4o-mini` (multimodal) | ✅ |
| Embeddings (Q8, Q13) | `text-embedding-3-small` | ✅ |
| Audio transcription (Q6) | `gemini-2.5-flash-lite` | ✅ (via geminiv1beta JSON API) |

> **Q6 has NO download button.** Unlike the other deploy questions, the Korean-audio
> grader generates the audio clips *server-side* and POSTs them to your `/answer-audio`
> endpoint at grading time. There's nothing to download — you just deploy the endpoint
> and submit its URL. Request body it receives: `{"audio_id":"...","audio_base64":"..."}`.

---

## 5. Verified worked examples (computed from real seed data)

These outputs were produced by running the guide's own scripts on actual downloaded
data. If your data matches, your script should print the same shape. **Your values
differ per email — use these only to confirm your script runs correctly.**

- **Q10** — token `b1fa45637742d1f2`, difficulty `27` → nonce **`28857739`**
  (found in ~84 s single-threaded). Confirms the miner + leading-zero-bit logic.
- **Q11** — the extractor pulled all 10 latest facts cleanly, e.g.
  `{"q1":"hybrid-rerank-v4","q3":"128","q4":"300","q8":"8:1","q10":"queue-meridian", ...}`
  (bare values, no "tokens" noise). This question is essentially free.
- **Q12** — deterministic `service→label` map over the 50 logs, sorted by id, gives a
  valid `classified.jsonl` and a stable `sha256`. Confirms the pipe recipe.
- **Q1 / Q5 / Q13** — scripts verified to run against the real JSON shapes
  (`doc_id`/`query_id`/`embedding` for Q5; `source_urls`/`limit` for Q1).

> Reminder: these exact strings are from one test seed. **Always compute your own.**

---

Now open **`PART-A.md`** and start with the easy, guaranteed marks.
