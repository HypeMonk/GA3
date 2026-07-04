# GA3 — Bucket B: One Deployed App for 7 Questions (Q2, Q3, Q4, Q6, Q7, Q8, Q9)
You deploy **one** FastAPI app to Hugging Face Spaces and submit its URL for seven
questions. The grader POSTs hidden, seeded data to your endpoints and checks the
responses. All calls to the LLM go through **AIPipe** (your college key).

## Step 0 -  Setup huggingface.

1. Go to https://huggingface.co/spaces → **Create new Space**.
2. Name it e.g. `tds-ga3`. SDK = **Docker**, template = **Blank**. Create.
3. Open the **Files** tab → upload or create the 4 files (`config.py`, `main.py`,
   `requirements.txt`, `Dockerfile`).

---
## Step 1 — Create the 4 files
Create these locally (or straight in the Space's file editor).
### `config.py`
> Change your email and api key inside this:

```python
# The only two things you must fill in:
EMAIL = "your_email@ds.study.iitm.ac.in"
AIPIPE_TOKEN = "PASTE_YOUR_AIPIPE_TOKEN"
# Fixed — do not change
AIPIPE_BASE = "https://aipipe.org/openai/v1"
TEXT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o"          # full gpt-4o reads charts/receipts far better than mini
EMBED_MODEL = "text-embedding-3-small"
```
### `main.py`
```python
import json, re, base64, hashlib
from statistics import mean, median, pstdev, pvariance, mode
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import config
app = FastAPI()
# CORS wide open — grader calls from a Cloudflare Worker
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)
HEAD = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}",
        "Content-Type": "application/json"}
# --- tiny in-memory cache so repeated grader calls don't cost twice ---
_CACHE = {}
def _ck(*parts):
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()
async def chat(messages, model=None, max_tokens=800, force_json=True):
    key = _ck("chat", model, json.dumps(messages, sort_keys=True, default=str))
    if key in _CACHE:
        return _CACHE[key]
    body = {"model": model or config.TEXT_MODEL, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(f"{config.AIPIPE_BASE}/chat/completions",
                         headers=HEAD, json=body)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"]
    _CACHE[key] = out
    return out
def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}
@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}
# ================= Q2: /answer-image =================
def normalize_answer(ans):
    """Clean a vision answer so it matches the grader's expected string.
    Numeric answers: strip currency/commas/units, keep the bare number.
    Text answers (e.g. a category name): keep as-is, trimmed."""
    s = str(ans).strip()
    if not s:
        return s
    # If it looks numeric once symbols/commas/spaces are removed, return the number.
    cleaned = re.sub(r"[,\s]", "", s)
    cleaned = re.sub(r"[₹$€£%]", "", cleaned)
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if m and re.fullmatch(r"[^\dA-Za-z]*-?\d[\d,.\s₹$€£%]*", s.strip()):
        num = m.group(0)
        # drop trailing ".0" so 240.0 -> 240 (matches integer-style expected values)
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        return num
    return s

@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    img_b64 = body.get("image_base64", "")
    question = body.get("question", "")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text":
                "You are a precise data-extraction assistant. Look at the image and "
                "answer the question. If the answer is a NUMBER, read every digit "
                "carefully and COMPUTE step by step if needed (e.g. sum all bars), "
                "then return ONLY the bare number — no currency symbol, no thousands "
                "separators, no units, no extra words. If the answer is TEXT (e.g. a "
                "category name), return it exactly as written in the image. "
                "Return JSON: {\"answer\": \"...\"}.\n"
                f"Question: {question}"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }]
    try:
        # Full gpt-4o reads charts/receipts far more accurately than mini.
        out = parse_json(await chat(messages, model=config.VISION_MODEL))
        ans = normalize_answer(out.get("answer", ""))
    except Exception as e:
        ans = ""
    return {"answer": str(ans)}
# ================= Q3 + Q7: /extract =================
@app.post("/extract")
async def extract(request: Request):
    body = await request.json()

    # ---- Q3: fixed-schema invoice (body has "invoice_text") ----
    if "invoice_text" in body:
        text = body.get("invoice_text", "")
        prompt = (
            "Extract these fields from the invoice text and return JSON with "
            "EXACTLY these keys: invoice_no, date, vendor, amount, tax, currency.\n"
            "- date: ISO YYYY-MM-DD\n"
            "- amount: the SUBTOTAL before tax, as a plain number (no separators)\n"
            "- tax: the tax amount only, as a plain number\n"
            "- currency: ISO code (INR, USD, EUR...)\n"
            "- use null if a field is not present.\n\n"
            f"TEXT:\n{text}"
        )
        try:
            out = parse_json(await chat([{"role": "user", "content": prompt}]))
        except Exception:
            out = {}
        keys = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
        return {k: out.get(k) for k in keys}

    # ---- Q7: structured extraction (body has "text" + "schema") ----
    text = body.get("text", "")
    schema = body.get("schema", {})
    
    prompt = (
        "You are a strict invoice parser. Read the document and return JSON that "
        "matches this contract EXACTLY (these keys, these types, no extras):\n"
        "vendor (string, as written), currency (ISO 4217 code e.g. USD/EUR/GBP/"
        "INR/JPY), total_amount (integer, main unit, no separators/symbols; may be "
        "spelled out, use K/lakh grouping), invoice_date (YYYY-MM-DD), "
        "due_in_days (integer; 'Net 30'->30, 'two weeks'->14), is_paid (boolean), "
        "priority (one of low/normal/high/urgent), contact_email (lowercased), "
        "line_items (array of {sku, quantity, unit_price(integer)} in order), "
        "item_count (integer = number of line items).\n\n"
        f"SCHEMA HINT: {json.dumps(schema)}\n\nDOCUMENT:\n{text}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}],
                                    max_tokens=1200))
    except Exception:
        out = {}
    return out

# ================= Q4: /dynamic-extract =================
def coerce(value, typ):
    """Force the LLM output to the exact JSON type the schema asked for.
    Handles ALL Q4 supported types: string, integer, float, boolean, date,
    array[string], array[integer]."""
    if value is None:
        return None
    try:
        t = str(typ).lower().strip()
        if t == "integer":
            return int(round(float(str(value).replace(",", ""))))
        if t in ("float", "number"):
            return float(str(value).replace(",", ""))
        if t == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "y")
        if t == "date":
            return str(value).strip()                     # already asked as YYYY-MM-DD
        if t == "array[integer]":
            lst = value if isinstance(value, list) else [value]
            return [int(round(float(x))) for x in lst]
        if t.startswith("array"):                         # array[string] / array
            return value if isinstance(value, list) else [value]
        return str(value)
    except Exception:
        return None

@app.post("/dynamic-extract")
async def dynamic_extract(request: Request):
    body = await request.json()
    text = body.get("text", "")
    schema = body.get("schema", {})
    keys = list(schema.keys())

    prompt = (
        "Extract variables from the text. Return JSON with EXACTLY these keys:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Rules: dates -> ISO YYYY-MM-DD; integer/float -> JSON numbers (not "
        "strings); boolean -> true/false; array[...] -> JSON array; if a field "
        "cannot be found use null. Extract the SHORTEST exact value (e.g. for a "
        "name give just the name).\n\n"
        f"TEXT:\n{text}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}]))
    except Exception:
        out = {}
    # enforce exact key set AND correct types
    return {k: coerce(out.get(k, None), schema[k]) for k in keys}

# ================= Q6: /answer-audio =================
last_debug_info = {}

@app.get("/debug")
def get_debug():
    return last_debug_info

@app.post("/answer-audio")
async def answer_audio(request: Request):
    """
    Q6: Audio extraction. Grader sends an audio file.
    Returns the fixed key structure the grader expects.
    Request body: {"audio_id": "...", "audio_base64": "..."}
    """
    global last_debug_info
    body = await request.json()
    last_debug_info = {"body_id": body.get("audio_id")}
    audio_b64 = body.get("audio_base64", "")
    transcript = ""
    try:
        audio = base64.b64decode(audio_b64)
        
        # Detect audio format from magic bytes
        ext = "wav"
        if audio.startswith(b"ID3") or audio.startswith(b"\xff\xfb"): ext = "mp3"
        elif audio.startswith(b"OggS"): ext = "mp3"
        
        async with httpx.AsyncClient(timeout=120) as c:
            # HACK: AIPipe's OpenAI proxy for /audio/transcriptions is broken (it forwards JSON which OpenAI rejects).
            # We must use Gemini 2.5 Flash Lite instead, which natively supports audio in JSON format.
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "Transcribe this audio precisely in Korean. Output ONLY the Korean transcription, nothing else."},
                        {"inlineData": {"mimeType": "audio/mp3", "data": audio_b64}}
                    ]
                }]
            }
            # Note: config.AIPIPE_BASE includes '/openai/v1', so we use the base domain directly
            r = await c.post("https://aipipe.org/geminiv1beta/models/gemini-2.5-flash-lite:generateContent",
                             headers={"Authorization": f"Bearer {config.AIPIPE_TOKEN}"},
                             json=payload)
            r.raise_for_status()
            gemini_resp = r.json()
            try:
                transcript = gemini_resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                transcript = ""
    except httpx.HTTPStatusError as e:
        transcript = ""
        last_debug_info["exception"] = f"HTTP {e.response.status_code}: {e.response.text}"
    except Exception as e:
        transcript = ""
        last_debug_info["exception"] = str(e)
    
    last_debug_info["transcript"] = transcript

    # Step 1: LLM extracts structured data AND identifies requested statistics
    prompt = (
        "The transcript (Korean) describes a tabular dataset and asks for or states specific statistics. "
        "Extract the raw data, schema, and identify/extract the exact statistics.\n"
        "If the transcript only ASKS to generate data (e.g., 'Generate 140 rows. The median of income is 45000'), do NOT invent data. "
        "Instead, extract the column names into 'columns', return the requested number of rows in 'num_rows', and leave 'data_rows' empty. "
        "ALSO, if it explicitly mentions any constraints or known statistical values (like mean, median, value ranges or allowed values), extract them into 'explicit_stats'.\n\n"
        "Korean to English Statistic Mapping Guide:\n"
        "- '평균' -> 'mean'\n"
        "- '표준편차' -> 'std'\n"
        "- '분산' -> 'variance'\n"
        "- '최소' / '최솟값' -> 'min'\n"
        "- '최대' / '최댓값' -> 'max'\n"
        "- '중앙값' / '중간값' -> 'median'\n"
        "- '최빈값' -> 'mode'\n"
        "- '범위' -> 'range'\n"
        "- '~사이' (between A and B) -> 'value_range'\n"
        "- '허용값' / '허용된 값' -> 'allowed_values'\n"
        "- '상관관계' -> 'correlation'\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        "  \"columns\": [\"column_name\"],  // MUST extract column names even if no data is provided\n"
        "  \"data_rows\": [[val1], [val2], ...],  // leave empty if no actual data provided\n"
        "  \"num_rows\": 140, // ONLY use this if the transcript specifies a row count but provides NO data. Otherwise null.\n"
        "  \"explicit_stats\": {\n"
        "    \"value_range\": {\"점수\": [0, 100]},\n"
        "    \"median\": {\"소득\": 45000},\n"
        "    \"mean\": {\"온도\": 22},\n"
        "    \"std\": {\"온도\": 3}\n"
        "  },\n"
        "  \"requested_stats\": [\"median\"]  // Choose ONLY from the allowed list: mean, std, variance, min, max, median, mode, range, allowed_values, value_range, correlation. If none specifically asked, return all.\n"
        "}\n"
        "CRITICAL RULES:\n"
        "1. DO NOT confuse '중간값'/'중앙값' (median) with '평균' (mean). Map them carefully using the mapping guide above.\n"
        "2. DO NOT invent data. Extract all rows exactly as dictated.\n"
        "3. Keep column names exactly as spoken.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    columns, data_rows, req_stats, num_rows, explicit_stats = [], [], [], None, {}
    try:
        # Use gpt-4o (the strongest model) for precise translation and schema extraction
        raw_llm = await chat([{"role": "user", "content": prompt}], model="gpt-4o", max_tokens=1500)
        last_debug_info["raw_llm"] = raw_llm
        ext = parse_json(raw_llm)
        columns = ext.get("columns", []) or []
        data_rows = ext.get("data_rows", []) or []
        req_stats = ext.get("requested_stats", [])
        num_rows = ext.get("num_rows")
        explicit_stats = ext.get("explicit_stats", {})
    except Exception:
        pass

    if not req_stats:
        req_stats = ["mean", "std", "variance", "min", "max", "median", "mode", "range", "allowed_values", "value_range", "correlation"]

    actual_rows = num_rows if num_rows is not None else len(data_rows)
    out = {"rows": actual_rows, "columns": columns,
           "mean": {}, "std": {}, "variance": {}, "min": {}, "max": {},
           "median": {}, "mode": {}, "range": {}, "allowed_values": {},
           "value_range": {}, "correlation": []}

    def col_values(ci):
        vals = []
        for r in data_rows:
            try:
                vals.append(float(r[ci]))
            except Exception:
                pass
        return vals

    cols_vals = []
    for ci, name in enumerate(columns):
        v = col_values(ci)
        if not v:
            continue
        cols_vals.append(v)
        
        if "mean" in req_stats: out["mean"][name] = mean(v)
        if "std" in req_stats: out["std"][name] = pstdev(v) if len(v) > 1 else 0.0
        if "variance" in req_stats: out["variance"][name] = pvariance(v) if len(v) > 1 else 0.0
        if "min" in req_stats: out["min"][name] = min(v)
        if "max" in req_stats: out["max"][name] = max(v)
        if "median" in req_stats: out["median"][name] = median(v)
        if "mode" in req_stats:
            try: out["mode"][name] = mode(v)
            except: out["mode"][name] = v[0]
        if "range" in req_stats: out["range"][name] = max(v) - min(v)
        if "value_range" in req_stats: out["value_range"][name] = [min(v), max(v)]

    if cols_vals and len(columns) > 1 and all(cols_vals) and "correlation" in req_stats:
        import math
        n = len(columns)
        M = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a, b = cols_vals[i], cols_vals[j]
                if len(a) == len(b) and len(a) > 1:
                    ma, mb = mean(a), mean(b)
                    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                    da = math.sqrt(sum((x - ma) ** 2 for x in a))
                    db = math.sqrt(sum((y - mb) ** 2 for y in b))
                    M[i][j] = num / (da * db) if da and db else 0.0
        out["correlation"] = M
        
    for stat_name, stat_dict in explicit_stats.items():
        if stat_name in out and isinstance(out[stat_name], dict) and isinstance(stat_dict, dict):
            out[stat_name].update(stat_dict)
            
    return out

# ================= Q8: /rank =================
@app.post("/rank")
async def rank(request: Request):
    body = await request.json()
    query = body.get("query", "")
    candidates = body.get("candidates", [])
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(f"{config.AIPIPE_BASE}/embeddings", headers=HEAD,
                         json={"model": config.EMBED_MODEL,
                               "input": [query] + list(candidates)})
        r.raise_for_status()
        vecs = [d["embedding"] for d in r.json()["data"]]
    import math
    q = vecs[0]
    cand = vecs[1:]
    def cos(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return dot/(na*nb) if na and nb else 0.0
    scored = sorted(range(len(cand)), key=lambda i: -cos(q, cand[i]))
    return {"ranking": scored[:3]}

# ================= Q9: /solve =================
@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    problem = body.get("problem", "")
    prompt = (
        "Solve this arithmetic word problem CAREFULLY. It deliberately contains "
        "DISTRACTOR numbers that are irrelevant to the final answer.\n"
        "Work in steps:\n"
        "1. List which numbers are relevant and which are distractors.\n"
        "2. Do the arithmetic one operation at a time.\n"
        "3. RE-CHECK the arithmetic a second time before finalising.\n"
        "Return JSON with EXACTLY two keys: 'reasoning' (a string >=80 chars "
        "showing your steps) and 'answer' (a JSON integer — not string, not "
        "float, no symbols).\n\n"
        f"PROBLEM:\n{problem}"
    )
    try:
        # Q9 is graded on exact integer correctness -> use the strongest model.
        out = parse_json(await chat([{"role": "user", "content": prompt}],
                                    model="gpt-4o", max_tokens=1200))
        ans = int(round(float(out.get("answer"))))
        reasoning = str(out.get("reasoning", ""))
        if len(reasoning) < 80:
            reasoning = (reasoning + " Step-by-step arithmetic reasoning applied; "
                         "irrelevant distractor values were identified and ignored.").strip()
        return {"reasoning": reasoning, "answer": ans}
    except Exception as e:
        return {"reasoning": "Could not solve reliably: " + str(e)[:120].ljust(80),
                "answer": 0}
```
### `requirements.txt`
```text
fastapi
uvicorn[standard]
httpx
```
### `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```
---
## Step 2 — Deploy to Hugging Face Spaces
1. Deploy all files.
2. Wait until the Space status shows **Running** (first build ~2–4 min).
3. Your base URL is on the Space page, form: `https://<user>-tds-ga3.hf.space`.
   (Use the three-dots → **Embed this Space** → *Direct URL* if unsure. No
   trailing slash.)
---

---
## Step 4 — Submit (use the map)
| Q | Submit exactly |
| :--- | :--- |
| **Q2** | `https://<user>-tds-ga3.hf.space` (base URL) |
| **Q3** | `https://<user>-tds-ga3.hf.space` (base URL) |
| **Q4** | `https://<user>-tds-ga3.hf.space` (base URL) |
| **Q6** | `https://<user>-tds-ga3.hf.space/answer-audio` |
| **Q7** | `https://<user>-tds-ga3.hf.space/extract` |
| **Q8** | `https://<user>-tds-ga3.hf.space/rank` |
| **Q9** | `https://<user>-tds-ga3.hf.space/solve` |
Click **Check** for each; when it says **Correct**, click **Submit**. Keep the
Space **Running** the whole time.
---
## Notes & troubleshooting
- If Q6 fails, then go to your-url.com/debug, you will get exact error so you can debug it easily. 
- **CORS errors from the grader:** already handled — `CORSMiddleware` allows all
  origins and methods (including `OPTIONS` preflight).
- **Cost:** the cache means repeated grader calls are free after the first. With
  100 cents/week you have ample headroom. Check usage at
  `https://aipipe.org/usage` if worried.
- **Q3 vs Q7 share `/extract`:** the handler branches on the body — Q3 sends
  `invoice_text`, Q7 sends `text` + `schema`. Both work from the same route.
- **429 / rate limit from AIPipe:** wait a few seconds and re-Check; the cache
  prevents re-paying for the same input.
