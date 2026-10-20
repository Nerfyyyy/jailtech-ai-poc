import re, hashlib, os, json, datetime, requests
from pathlib import Path
import numpy as np
import faiss
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import gradio as gr

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
EMBED_MODEL = "all-MiniLM-L6-v2"
CC_CHUNK = 250
CC_OVERLAP = 50
STD_CHUNK = 600
STD_OVERLAP = 100
TOP_K = 3
GROQ_KEY = os.environ.get("GROQ_API_KEY", "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")
CORPUS_DIR = Path("corpus")
CORPUS_DIR.mkdir(exist_ok=True)

DOC_URLS = {
    "BJMP-ODCA-MC-167": "https://barmm.bjmp.gov.ph/wp-content/uploads/2025/05/03.31.2025-BJMP-ODCA-MC-167-CITIZENS-CHARTER.pdf",
    "BJMP-COM-2015": "https://car.bjmp.gov.ph/wp-content/uploads/2024/11/BJMP-Comprehensive-Operations-Manual-2015-Edition.pdf",
    "BJMP-FOI-2021": "https://barmm.bjmp.gov.ph/wp-content/uploads/2022/04/FOI-Manual-Sept-10-2021-FINAL.pdf",
    "BJMP-LSO-MC-124": "https://bjmp.gov.ph/downloadable-manuals/",
    "BJMP-DICTM-MC-128": "https://sites.google.com/bjmp.gov.ph/ictmdjsioc/dictm-memorandum-circular",
    "BJMP-DPRM-MC-73": "https://bjmp.gov.ph/policies/",
    "BJMP-CDS-MC-133": "https://www.scribd.com/document/823657970/02-21-2022-Mc-on-Communications-Policy",
}

CORPUS_DEFS = [
    {"file": "citizens_charter.pdf", "url": DOC_URLS["BJMP-ODCA-MC-167"], "title": "BJMP Citizen's Charter 2025", "doc_no": "BJMP-ODCA-MC-167", "date": "2025", "chunk": CC_CHUNK, "overlap": CC_OVERLAP},
    {"file": "operations_manual.pdf", "url": DOC_URLS["BJMP-COM-2015"], "title": "BJMP Comprehensive Operations Manual 2015", "doc_no": "BJMP-COM-2015", "date": "2015", "chunk": STD_CHUNK, "overlap": STD_OVERLAP},
    {"file": "foi_manual.pdf", "url": DOC_URLS["BJMP-FOI-2021"], "title": "BJMP Freedom of Information Manual", "doc_no": "BJMP-FOI-2021", "date": "2021", "chunk": STD_CHUNK, "overlap": STD_OVERLAP},
]

# ═══════════════════════════════════════════
# DOWNLOAD PDFs AT STARTUP
# ═══════════════════════════════════════════
def download_pdf(url, dest):
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return True
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 10000:
            dest.write_bytes(r.content)
            print(f"  [downloaded] {dest.name} ({len(r.content)//1024} KB)")
            return True
    except Exception as e:
        print(f"  [failed] {dest.name}: {e}")
    return False

print("═══ DOWNLOADING CORPUS ═══")
for cd in CORPUS_DEFS:
    download_pdf(cd["url"], CORPUS_DIR / cd["file"])

# ═══════════════════════════════════════════
# EXTRACT + CHUNK
# ═══════════════════════════════════════════
def clean_text(txt):
    txt = txt.replace("\u00ad", "")
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()

def extract_pdf(path):
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            txt = clean_text(pg.extract_text() or "")
            if txt:
                pages.append({"page": i, "text": txt})
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars > 2000:
        return pages, "text"
    print("    → Using OCR...")
    ocr_pages = convert_from_path(str(path), dpi=200)
    results = []
    for i, page in enumerate(ocr_pages, start=1):
        txt = clean_text(pytesseract.image_to_string(page))
        if txt and len(txt) > 20:
            results.append({"page": i, "text": txt})
    return results, "ocr"

def approx_tokens(text):
    return len(text.split())

def make_chunk(text, meta, pages):
    page_label = f"p.{pages[0]}" if len(set(pages)) == 1 else f"pp.{min(pages)}-{max(pages)}"
    cid = hashlib.sha1(f"{meta['doc_no']}|{page_label}|{text[:120]}".encode()).hexdigest()[:12]
    return {"chunk_id": cid, "doc_title": meta["title"], "doc_no": meta["doc_no"],
            "doc_date": meta.get("date",""), "section": page_label, "text": text}

def chunk_pages(pages, meta, size, overlap):
    ch = []
    step = size - overlap
    if step <= 0:
        step = size // 2
    buffer, buffer_pages = [], []
    for page in pages:
        buffer.append(page["text"])
        buffer_pages.append(page["page"])
        joined = "\n".join(buffer)
        while approx_tokens(joined) >= size:
            words = joined.split()
            window = " ".join(words[:size])
            ch.append(make_chunk(window, meta, buffer_pages))
            remaining = " ".join(words[step:])
            joined = remaining
            buffer = [remaining]
    if buffer and approx_tokens("\n".join(buffer)) > 20:
        ch.append(make_chunk("\n".join(buffer), meta, buffer_pages))
    return ch

print("\n═══ EXTRACTING & CHUNKING ═══")
all_chunks = []
for cd in CORPUS_DEFS:
    f = CORPUS_DIR / cd["file"]
    if not f.exists():
        print(f"  SKIP (not downloaded): {cd['title']}")
        continue
    pages, method = extract_pdf(f)
    if not pages:
        print(f"  SKIP (no text): {cd['title']}")
        continue
    doc_chunks = chunk_pages(pages, cd, cd["chunk"], cd["overlap"])
    all_chunks.extend(doc_chunks)
    print(f"  ✓ {cd['title']}: {len(pages)} pages → {len(doc_chunks)} chunks [{method}]")

print(f"\nTotal chunks: {len(all_chunks)}")

# ═══════════════════════════════════════════
# BUILD INDEX
# ═══════════════════════════════════════════
print("\n═══ BUILDING INDEX ═══")
retriever_model = SentenceTransformer(EMBED_MODEL)
embeddings = retriever_model.encode([c["text"] for c in all_chunks], normalize_embeddings=True, show_progress_bar=True)
embeddings = np.asarray(embeddings, dtype="float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
chunks = all_chunks
print(f"Index built: {index.ntotal} vectors")

# ═══════════════════════════════════════════
# GROQ
# ═══════════════════════════════════════════
def generate_groq(prompt):
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 512, "temperature": 0.1},
            timeout=30
        )
        if r.status_code != 200:
            return f"[API Error {r.status_code}: {r.json().get('error',{}).get('message','Unknown')}]"
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error: {str(e)}]"

# ═══════════════════════════════════════════
# QUERY REWRITING
# ═══════════════════════════════════════════
def rewrite_query(query, history):
    if not history or len(history) < 2:
        return query
    context = ""
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content'][:200]}\n"
    prompt = f"""Given this conversation about BJMP administrative policy:
{context}
The user's latest message is: "{query}"
Rewrite the user's latest message as a standalone, self-contained question. Do NOT answer it. Output only the rewritten question on one line.
If already self-contained, return it unchanged."""
    try:
        rewritten = generate_groq(prompt).strip().strip('"').strip("'")
        return rewritten if rewritten else query
    except:
        return query

# ═══════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════
def format_citation(doc_title, doc_no, section, score):
    url = DOC_URLS.get(doc_no, "#")
    label = f"{doc_title} ({doc_no}), {section} ({score:.2f})"
    if url != "#":
        return f"- [{label}]({url})"
    return f"- {label}"

def format_confidence_bar(score):
    filled = int(score * 20)
    return "█" * filled + "░" * (20 - filled) + f" {score:.2f}"

AUDIT_FILE = "audit_log.json"

def save_audit(entry):
    try:
        with open(AUDIT_FILE) as f:
            log = json.load(f)
    except:
        log = []
    log.append(entry)
    with open(AUDIT_FILE, "w") as f:
        json.dump(log, f, indent=2)

def chat_fn(query, history):
    rewritten = rewrite_query(query, history)
    q = retriever_model.encode([rewritten], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, idxs = index.search(q, TOP_K)
    hits = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        hits.append({"score": float(score), "chunk": chunks[i]})
    passages = []
    for n, h in enumerate(hits, start=1):
        c = h["chunk"]
        passages.append(f"[Passage {n}]\nDocument: {c['doc_title']} ({c['doc_no']}, {c['doc_date']})\nLocation: {c['section']}\nText: {c['text']}\n")
    context = ""
    if history:
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content'][:300]}\n"
    system_prompt = """You are a helpful administrative assistant for the Philippine Bureau of Jail Management and Penology (BJMP).
Tone: Warm, clear, professional. Speak naturally. Use "you" and "I" freely. Light conversational phrasing is fine but avoid slang.
Rules:
- Answer ONLY using the provided passages. Every factual claim must be traceable to a passage.
- Cite the document number and section for factual claims.
- If the passages do not contain the answer, say: "I don't have that information in the documents I can access. I'd recommend referring this to the appropriate office."
- Never guess, infer, or draw on outside knowledge.
- Keep answers concise. Use short paragraphs or bullet points.
- If the user asks a follow-up, acknowledge it briefly before answering.
- If the user greets you or says something off-topic, respond briefly and redirect: "I'm here to help with BJMP administrative policy questions. How can I assist?"
- Do not provide legal advice or official policy interpretations."""
    prompt = system_prompt
    if context:
        prompt += f"\n\n--- CONVERSATION SO FAR ---\n{context}"
    prompt += f"\n--- PASSAGES ---\n" + "\n".join(passages) + f"\n\nQ: {query}\nA:"
    answer = generate_groq(prompt)
    SENSITIVITY = ["pdl", "persons deprived of liberty", "inmate", "detainee",
                   "personnel record", "201 file", "service record",
                   "intelligence", "case file", "criminal record",
                   "single carpeta", "password", "classified"]
    reasons = []
    if not hits:
        reasons.append("No passages retrieved")
    if hits and max(h["score"] for h in hits) < 0.35:
        reasons.append("Similarity below threshold")
    if "I don't have that information" in answer:
        reasons.append("System refused")
    if any(kw in query.lower() for kw in SENSITIVITY):
        reasons.append("Sensitivity keyword")
    route = "HUMAN_REVIEW" if reasons else "AUTO"
    save_audit({"timestamp": datetime.datetime.now().isoformat(), "query": query, "rewritten": rewritten, "route": route, "reasons": reasons, "top_scores": [round(h["score"],4) for h in hits]})
    display = ""
    if route == "HUMAN_REVIEW":
        display += "⚠️ *Routed to Human Review* — " + ", ".join(reasons) + "\n\n"
    display += answer
    if hits:
        display += "\n\n**CONFIDENCE:**\n"
        for h in hits[:3]:
            c = h["chunk"]
            display += f"`{format_confidence_bar(h['score'])}` {c['doc_title']}\n"
        display += "\n**SOURCES:**\n"
        for h in hits[:3]:
            c = h["chunk"]
            display += format_citation(c["doc_title"], c["doc_no"], c["section"], h["score"]) + "\n"
    if route == "HUMAN_REVIEW":
        display += "\n---\n*Flagged for human verification.*"
    return display

# ═══════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════
print("\n═══ LAUNCHING ═══")
demo = gr.Blocks(title="JailTech AI — RAG PoC", theme=gr.themes.Soft())
with demo:
    gr.Markdown("# 🏛️ JailTech AI — RAG Proof of Concept")
    gr.Markdown("*Non-evaluative demo. All outputs require human verification. No PDL, personnel, or intelligence content.*")
    gr.ChatInterface(fn=chat_fn, title="BJMP Administrative Policy Assistant", description="Ask questions and follow up naturally.")
demo.launch(server_name="0.0.0.0", server_port=7860)   
