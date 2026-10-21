import re, hashlib, os, json, datetime, requests
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
GROQ_KEY = os.environ.get("GROQ_API_KEY", "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")

# ═══════════════════════════════════════════
# LOAD PRE-BUILT INDEX
# ═══════════════════════════════════════════
print("Loading pre-built index...")
index = faiss.read_index("jailtech.index")
with open("chunks.json") as f:
    chunks = json.load(f)
print(f"Index loaded: {index.ntotal} vectors, dim={index.d}")

retriever_model = SentenceTransformer(EMBED_MODEL)
print("Embedding model loaded.")

# ═══════════════════════════════════════════
# CONTACT INFO (verified from bjmp.gov.ph)
# ═══════════════════════════════════════════
CONTACT_INFO = """
---
**For further inquiries:**

| Office | Trunk Line | Email |
|--------|-----------|-------|
| BJMP NHQ (Quezon City) | (02) 8927-6383, 8926-7756, 8927-5505, 8927-6154 | [cdsmessagecenter@bjmp.gov.ph](mailto:cdsmessagecenter@bjmp.gov.ph) |
| BJMP Region IV-A (CALABARZON) | +63-932-883-8754 | [bjmpro4a@bjmp.gov.ph](mailto:bjmpro4a@bjmp.gov.ph) |
| Region I (Cagayan Valley) | (072) 619-5493; 0920-590-463 | [r1.messagecenter@bjmp.gov.ph](mailto:r1.messagecenter@bjmp.gov.ph) |
| Region V (Bicol) | 0948-318-2616; 0927-478-3212 | [bjmpro5@bjmp.gov.ph](mailto:bjmpro5@bjmp.gov.ph) |
| Region VI (Western Visayas) | (033) 338-2847 | — |
| Region IX (Western Mindanao) | (062) 925-3979 | [bjmpro9@bjmp.gov.ph](mailto:bjmpro9@bjmp.gov.ph) |

*For other regional offices and jail units, visit: [bjmp.gov.ph/facilities/list-of-jails](https://bjmp.gov.ph/facilities/list-of-jails)*
"""

# ═══════════════════════════════════════════
# DOCUMENT URLs
# ═══════════════════════════════════════════
DOC_URLS = {
    "BJMP-ODCA-MC-167": "https://barmm.bjmp.gov.ph/wp-content/uploads/2025/05/03.31.2025-BJMP-ODCA-MC-167-CITIZENS-CHARTER.pdf",
    "BJMP-COM-2015": "https://car.bjmp.gov.ph/wp-content/uploads/2024/11/BJMP-Comprehensive-Operations-Manual-2015-Edition.pdf",
    "BJMP-FOI-2021": "https://barmm.bjmp.gov.ph/wp-content/uploads/2022/04/FOI-Manual-Sept-10-2021-FINAL.pdf",
    "BJMP-LSO-MC-124": "https://bjmp.gov.ph/downloadable-manuals/",
    "BJMP-DICTM-MC-128": "https://sites.google.com/bjmp.gov.ph/ictmdjsioc/dictm-memorandum-circular",
    "BJMP-DPRM-MC-73": "https://bjmp.gov.ph/policies/",
    "BJMP-CDS-MC-133": "https://www.scribd.com/document/823657970/02-21-2022-Mc-on-Communications-Policy",
    "BJMP-DPRM-MC-034": "https://bjmp.gov.ph/policies/",
    "BJMP-DO-MC-038": "https://bjmp.gov.ph/policies/",
    "BJMP-DICTM-MC-161": "https://car.bjmp.gov.ph/policies/",
    "BJMP-DHRD-MC-117": "https://bjmp.gov.ph/policies/",
    "BJMP-MSJ-2024": "https://calabarzon.bjmp.gov.ph/downloadable-manuals/",
}

# ═══════════════════════════════════════════
# SUGGESTED FOLLOW-UPS
# ═══════════════════════════════════════════
SUGGESTIONS = {
    "BJMP-ODCA-MC-167": ["What are the requirements for this service?", "What is the processing time?", "Where do I submit the request?"],
    "BJMP-COM-2015": ["What is the organizational structure?", "Who reports to the Director?", "What are the duties of this position?"],
    "BJMP-FOI-2021": ["How long does it take?", "What if my request is denied?", "Who is the FOI Officer?"],
    "BJMP-LSO-MC-124": ["What are the grounds for disciplinary action?", "What is the appeal process?", "Who sits on the disciplinary board?"],
    "BJMP-DICTM-MC-128": ["What are the acceptable use rules?", "What is the retention period?", "Who administers the e-mail system?"],
    "BJMP-DPRM-MC-73": ["What is the record-keeping procedure?", "How are transfers processed?", "Who has access to personnel records?"],
    "BJMP-CDS-MC-133": ["What is the document routing procedure?", "How are inter-office memos handled?", "What is the classification level?"],
    "BJMP-DPRM-MC-034": ["What are the ranks in order?", "What is the equivalent of this rank?", "How is promotion handled?"],
    "BJMP-DO-MC-038": ["What services do the help desks provide?", "What are the operating hours?", "How do I access the help desk?"],
    "BJMP-DICTM-MC-161": ["Who can use digital signatures?", "What documents require a digital signature?", "How is identity verified?"],
    "BJMP-DHRD-MC-117": ["What training programs are available?", "What is the scholarship process?", "What is the service obligation?"],
    "BJMP-MSJ-2024": ["What are the minimum space requirements?", "What are the ventilation standards?", "What are the fire safety requirements?"],
}

# ═══════════════════════════════════════════
# META-QUESTION HANDLER
# ═══════════════════════════════════════════
META_RESPONSES = {
    "search engine": "This system is more convenient than a search engine in three ways: (1) It reads the documents for you and synthesizes one grounded answer instead of giving you 10 PDF links to read manually. (2) It remembers context between turns, so you can ask follow-ups without repeating the full question. (3) It refuses to answer what it cannot verify and routes sensitive queries to a human — a search engine will happily return irrelevant or outdated results. For urgency context, please refer to the study's Introduction section.",
    "urgent": "The urgency of this system is grounded in the administrative workload and query volume of BJMP facilities, as discussed in the study's Introduction. This prototype demonstrates architectural feasibility; the empirical justification for urgency is presented in Chapter One of the thesis.",
}

# ═══════════════════════════════════════════
# GROQ
# ═══════════════════════════════════════════
def generate_groq(prompt):
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-oss-20b",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 512, "temperature": 0.1},
            timeout=30
        )
        if r.status_code != 200:
            return f"[API Error {r.status_code}: {r.json().get('error',{}).get('message','Unknown')}]"
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "[Error: Timed out. Please try again.]"
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
# CITATION FORMATTER
# ═══════════════════════════════════════════
def format_citation(doc_title, doc_no, section, score):
    url = DOC_URLS.get(doc_no, "#")
    page_match = re.search(r'p+\.?(\d+)', section)
    if url != "#" and page_match:
        page_num = page_match.group(1)
        if url.endswith(".pdf"):
            url += f"#page={page_num}"
    label = f"{doc_title} ({doc_no}), {section} ({score:.2f})"
    if url != "#":
        return f"- [{label}]({url})"
    return f"- {label}"

def format_confidence_bar(score):
    filled = int(score * 20)
    return "█" * filled + "░" * (20 - filled) + f" {score:.2f}"

# ═══════════════════════════════════════════
# SUGGESTED FOLLOW-UPS
# ═══════════════════════════════════════════
def get_suggestions(hits):
    if not hits:
        return ["What is the Freedom of Information procedure in BJMP?",
                "What services are available under the BJMP Citizen's Charter?",
                "What is the organizational structure of BJMP?"]
    top_doc = hits[0]["chunk"]["doc_no"]
    return SUGGESTIONS.get(top_doc, ["What are the requirements?", "How long does it take?", "Who do I contact?"])

# ═══════════════════════════════════════════
# META-QUESTION CHECK
# ═══════════════════════════════════════════
def check_meta(query):
    q_lower = query.lower()
    for key, response in META_RESPONSES.items():
        if key in q_lower:
            return response
    return None

# ═══════════════════════════════════════════
# CHAT FUNCTION
# ═══════════════════════════════════════════
def chat_fn(query, history):
    meta = check_meta(query)
    if meta:
        return meta + "\n" + CONTACT_INFO

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

Tone: Warm, clear, professional. Speak naturally. Use "you" and "I" freely. Light conversational phrasing is fine ("I see," "Here's what I found") but avoid slang.

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
    suggestions = get_suggestions(hits)
    display += "\n\n**Suggested follow-ups:**\n"
    for s in suggestions:
        display += f"- {s}\n"
    display += CONTACT_INFO
    return display

# ═══════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════
print("Launching Gradio...")

BG_CSS = """
<style>
#bjmp-header {
    background-image: linear-gradient(rgba(0, 20, 40, 0.7), rgba(0, 20, 40, 0.7)), url('https://i.imgur.com/Q1vdUii.png');
    background-size: cover;
    background-position: center top;
    padding: 30px 20px;
    border-radius: 10px;
    margin-bottom: 15px;
    text-align: center;
}
#bjmp-header h1 {
    color: #ffffff;
    font-size: 1.6em;
    margin-bottom: 5px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.7);
}
#bjmp-header p {
    color: #e0e0e0;
    font-size: 0.9em;
    margin: 0;
    text-shadow: 0 1px 3px rgba(0,0,0,0.7);
}
</style>
"""

demo = gr.Blocks(title="JailTech AI — RAG PoC", theme=gr.themes.Soft(), head=BG_CSS)
with demo:
    gr.HTML('<div id="bjmp-header"><h1>🏛️ JailTech AI — RAG Proof of Concept</h1><p>BJMP Administrative Policy Assistant | Non-Evaluative Demo | All outputs require human verification</p></div>')
    gr.ChatInterface(fn=chat_fn,
                     title="BJMP Administrative Policy Assistant",
                     description="Ask questions and follow up naturally.",
                     examples=[
                         "What is the Freedom of Information procedure in BJMP?",
                         "What services are available under the BJMP Citizen's Charter?",
                         "What is the organizational structure of BJMP?",
                         "What is the detention status of PDL Juan Dela Cruz?",
                         "How is this more convenient than a search engine?",
                     ])
demo.launch(server_name="0.0.0.0", server_port=7860)   
