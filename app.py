
import streamlit as st
import json
import re
import html
import time
import fitz
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PDF READER + RAG + GEMINI
# ==========================================

def extract_pdf_text(uploaded_file):
    pdf = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    pages = []
    for page_number, page in enumerate(pdf, start=1):
        page_text = page.get_text("text")
        if page_text.strip():
            pages.append(f"\n\n[PDF PAGE {page_number}]\n{page_text}")

    return "\n".join(pages)


def split_into_chunks(text, chunk_size=1600, overlap=220):
    text = " ".join(text.split())

    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def retrieve_relevant_chunks(question, chunks, top_k=6):
    if not chunks:
        return []

    if len(chunks) == 1:
        return chunks

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:
        matrix = vectorizer.fit_transform(
            chunks + [question]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        ).flatten()

        best_indices = scores.argsort()[-top_k:][::-1]

        return [
            chunks[i]
            for i in best_indices
            if scores[i] > 0
        ]

    except ValueError:
        return chunks[:top_k]


def ask_gemini(question, context, topic):
    """Use a fast Gemini Flash-Lite request shared by all study features."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Add GEMINI_API_KEY to .streamlit/secrets.toml."
        )

    # Do not silently truncate source material. Long-document features split it into batches.
    context = context or ""
    if context:
        prompt = f"""You are StudyMate AI, a friendly personal study tutor.
Use the study material below to answer the student's request.

STUDENT'S TOPIC: {topic or "General learning"}
STUDY MATERIAL:
{context}

STUDENT'S REQUEST:
{question}

Use simple student-friendly language. Follow any requested format exactly.
Use headings and concise bullet points where useful. Give examples when useful.
Use the PDF as the primary source. If the PDF does not contain enough information, answer using reliable general knowledge, and clearly label that part as “Additional explanation (not stated in the PDF)”. Never pretend extra information came from the PDF; acknowledge uncertainty when needed."""
    else:
        prompt = f"""You are StudyMate AI, a friendly personal study tutor.
Student's topic: {topic or "General learning"}
Student's request: {question}
Answer the question directly in simple language, even when it is outside the uploaded PDF. Use reliable general knowledge when needed, and label extra information separately from PDF-based information. Follow the requested format."""

    # Flash-Lite is Google's faster, lower-latency model for routine tasks.
    # One HTTP attempt prevents the SDK's long default retry waits.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=60000,
          retry_options=types.HttpRetryOptions(attempts=4),
        ),
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=3000,
                temperature=0.4,
            ),
        )
        answer = getattr(response, "text", None)
        if not answer or not answer.strip():
            raise RuntimeError("Gemini returned an empty answer. Please try again.")
        return answer.strip()
    except Exception as error:
        # Preserve the real Google error so we can distinguish overload, quota,
        # invalid credentials, model access, and network problems.
        detail = str(error).strip().replace("\n", " ")
        if len(detail) > 500:
            detail = detail[:500] + "..."
        lower = detail.lower()
        if "401" in lower or "unauthenticated" in lower or "api key" in lower:
            message = "Gemini rejected the API key. Check GEMINI_API_KEY in .streamlit/secrets.toml."
        elif "403" in lower or "permission_denied" in lower:
            message = "Gemini API access is not enabled for this key or project. Check Google AI Studio API access."
        elif "404" in lower or "not_found" in lower or "model" in lower and "not found" in lower:
            message = "The selected Gemini model is not available to this API key."
        elif "429" in lower or "resource_exhausted" in lower or "quota" in lower:
            message = "Gemini rate limit or quota reached. Wait for the limit to reset, then try again."
        elif "503" in lower or "unavailable" in lower or "overload" in lower:
            message = "Google's Gemini service is temporarily overloaded. This is not an app layout or code error. Try again in a little while."
        elif "timeout" in lower or "timed out" in lower or "504" in lower or "deadline" in lower:
            message = "Gemini did not respond within 60 seconds. Try a shorter question or a smaller PDF."
        else:
            message = "Gemini request failed. Check the technical detail below."
        raise RuntimeError(f"{message}\n\nTechnical detail: {detail or type(error).__name__}") from error


def get_study_context():
    """Return the complete extracted PDF text; never silently keep only its first pages."""
    if st.session_state.study_file:
        text = extract_pdf_text(st.session_state.study_file)
        if not text.strip():
            raise ValueError("No selectable text was found in this PDF. It may be scanned; an OCR version is needed.")
        return text
    return st.session_state.study_topic.strip()


def split_document_batches(text, batch_chars=16000):
    """Split the entire document into ordered batches without dropping later pages."""
    text = text or ""
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    batches, current = [], ""
    for paragraph in paragraphs:
        # Break unusually long paragraphs into safe-sized pieces.
        pieces = [paragraph[i:i + batch_chars] for i in range(0, len(paragraph), batch_chars)]
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > batch_chars:
                batches.append(current)
                current = piece
            else:
                current = (current + "\n\n" + piece).strip()
    if current:
        batches.append(current)
    return batches


def generate_full_document_summary(style, full_text):
    """Summarize every batch of a long PDF, retaining the document's full lesson coverage."""
    batches = split_document_batches(full_text)
    if not batches:
        raise ValueError("There is no readable text to summarize.")
    sections = []
    total = len(batches)
    for index, batch in enumerate(batches, start=1):
        prompt = (
            f"Create part {index} of {total} of a {style} study summary. "
            "Cover every important point in THIS section of the source. Include definitions, "
            "subtopics, explanations, examples, formulas/steps, and exam-relevant details where present. "
            "Do not skip content just to be short. Do not add unsupported facts. "
            "Use clear Markdown headings and bullets. This is one part of a larger lesson; "
            "do not write a final conclusion unless the source section has one."
        )
        with st.spinner(f"📖 Reading the full PDF — preparing section {index} of {total}..."):
            sections.append(generate_feature_content(prompt, batch, task_type="summary"))
    if len(sections) == 1:
        return sections[0]
    return "\n\n---\n\n".join(
        f"## Lesson section {i}\n\n{content}" for i, content in enumerate(sections, start=1)
    )


def generate_full_document_flashcards(topic, count, full_text):
    """Generate cards across the whole PDF by allocating cards to every document batch."""
    batches = split_document_batches(full_text)
    if not batches:
        raise ValueError("There is no readable text to create flashcards from.")
    # Distribute cards across the entire document. If there are more batches
    # than cards, sample evenly from the first through the final batch.
    if len(batches) > count:
        if count == 1:
            selected = [batches[len(batches) // 2]]
        else:
            indices = [round(i * (len(batches) - 1) / (count - 1)) for i in range(count)]
            selected = [batches[i] for i in indices]
        batches = selected
    allocations = [count // len(batches)] * len(batches)
    for i in range(count % len(batches)):
        allocations[i] += 1
    cards = []
    for index, (batch, amount) in enumerate(zip(batches, allocations), start=1):
        if amount <= 0:
            continue
        prompt = (
            f'Create exactly {amount} concise study flashcards about {topic or "the uploaded lesson"} '
            f'from section {index} of {len(batches)} of the PDF. Cover the most important distinct concepts. '
            'Return ONLY valid JSON in this exact structure: '
            '{"flashcards":[{"question":"...","answer":"..."}]}. '
            'Keep each question and answer readable and accurate. No Markdown fences or extra text.'
        )
        raw = generate_feature_content(prompt, batch, task_type="flashcards")
        data = parse_json_response(raw)
        cards.extend(data.get("flashcards", []))
    return cards[:count]

# ==========================================
# LANGGRAPH MULTI-AGENT STUDY WORKFLOW
# ==========================================

class StudyState(TypedDict, total=False):
    """Shared state passed between StudyMate's specialised agent nodes."""
    question: str
    topic: str
    uploaded_file: Any
    context: str
    cleaned_context: str
    chunks: list
    prepared_context: str
    learner_profile: str
    answer: str
    task_type: str
    quality_check: str
    selected_agents: list
    quiz_history: list


def orchestrator_agent(state: StudyState):
    """Coordinator: selects the workflow stages needed for the requested study task."""
    task_type = state.get("task_type", "general")
    agents = [
        "Data Cleaning Agent",
        "Data Training & Knowledge Index Agent",
        "Learner Evolution Agent",
        "Explanation Agent",
        "Data Testing Agent",
    ]
    return {"selected_agents": agents, "task_type": task_type}


def data_cleaning_agent(state: StudyState):
    """Extract PDF text when supplied, then clean and normalise the study content."""
    source_text = state.get("context", "") or ""
    uploaded_file = state.get("uploaded_file")
    if uploaded_file is not None:
        source_text = extract_pdf_text(uploaded_file)
        if not source_text.strip():
            raise ValueError("No selectable text was found in this PDF. It may be scanned; please use a searchable/OCR PDF.")
    cleaned = source_text.replace("\x00", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return {"cleaned_context": cleaned}


def data_training_agent(state: StudyState):
    """Build a fitted TF-IDF knowledge index from cleaned notes.

    This trains the document retrieval vectorizer/index; it does not fine-tune Gemini.
    """
    cleaned = state.get("cleaned_context", "") or ""
    chunks = split_into_chunks(cleaned) if cleaned else []
    prepared_context = cleaned
    task_type = state.get("task_type", "general")

    if chunks:
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            document_matrix = vectorizer.fit_transform(chunks)
            # For direct questions, retrieve only the most relevant note sections.
            # For summaries, quizzes, flashcards and plans, preserve the full supplied batch.
            if task_type == "ask_ai":
                query_vector = vectorizer.transform([state.get("question", "")])
                scores = cosine_similarity(query_vector, document_matrix).flatten()
                best_indices = scores.argsort()[-6:][::-1]
                selected = [chunks[i] for i in best_indices if scores[i] > 0]
                prepared_context = "\n\n".join(selected) if selected else ""
        except ValueError:
            prepared_context = "\n\n".join(chunks[:6]) if task_type == "ask_ai" else cleaned

    return {"chunks": chunks, "prepared_context": prepared_context}


def learner_evolution_agent(state: StudyState):
    """Analyse quiz history and prepare a lightweight learner profile for personalisation."""
    history = state.get("quiz_history", []) or []
    if not history:
        profile = "No quiz history is available yet. Teach at the student's current level and explain concepts clearly."
    else:
        topic_scores = {}
        for item in history:
            total = max(int(item.get("total", 0)), 1)
            score = int(item.get("score", 0))
            topic = str(item.get("topic", "General learning"))
            topic_scores.setdefault(topic, []).append(score / total)
        summaries = []
        for topic, scores in topic_scores.items():
            average = sum(scores) / len(scores)
            label = "needs more practice" if average < 0.6 else "developing" if average < 0.8 else "strong"
            summaries.append(f"{topic}: {average:.0%} average ({label})")
        profile = "Learner performance from this session: " + "; ".join(summaries)
    return {"learner_profile": profile}


def explanation_agent(state: StudyState):
    """Generate the requested learning output with Gemini using prepared notes and learner context."""
    question = state.get("question", "")
    topic = state.get("topic", "")
    context = state.get("prepared_context", "") or ""
    profile = state.get("learner_profile", "")
    enriched_request = (
        f"{question}\n\nLEARNER PROFILE FOR PERSONALISATION:\n{profile}\n\n"
        "Adapt the explanation and examples to this learner where relevant. "
        "Do not invent performance data or mention this profile unless useful."
    )
    answer = ask_gemini(enriched_request, context, topic)
    return {"answer": answer}


def data_testing_agent(state: StudyState):
    """Check that generated content is present and structured outputs contain usable data."""
    answer = (state.get("answer", "") or "").strip()
    if not answer:
        raise ValueError("The Explanation Agent returned an empty response. Please try again.")

    task_type = state.get("task_type", "general")
    if task_type in ("quiz", "flashcards"):
        data = parse_json_response(answer)
        key = "questions" if task_type == "quiz" else "flashcards"
        items = data.get(key, [])
        if not isinstance(items, list) or not items:
            raise ValueError(f"The Data Testing Agent could not find valid {key} in the AI response. Please generate again.")
        for number, item in enumerate(items, start=1):
            if task_type == "quiz":
                options = item.get("options", [])
                correct = str(item.get("answer", "")).strip()
                if not item.get("question") or len(options) != 4 or correct not in [str(option).strip() for option in options]:
                    raise ValueError(f"Quiz question {number} failed validation. Please generate the quiz again.")
            elif not item.get("question") or not item.get("answer"):
                raise ValueError(f"Flashcard {number} is missing a question or answer. Please generate again.")

    return {"quality_check": "Passed: response is non-empty and structured output checks succeeded."}


def create_study_graph():
    """LangGraph orchestrates the specialised agents in a tested study workflow."""
    workflow = StateGraph(StudyState)
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("data_cleaning", data_cleaning_agent)
    workflow.add_node("data_training", data_training_agent)
    workflow.add_node("learner_evolution", learner_evolution_agent)
    workflow.add_node("explanation", explanation_agent)
    workflow.add_node("data_testing", data_testing_agent)

    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "data_cleaning")
    workflow.add_edge("data_cleaning", "data_training")
    workflow.add_edge("data_training", "learner_evolution")
    workflow.add_edge("learner_evolution", "explanation")
    workflow.add_edge("explanation", "data_testing")
    workflow.add_edge("data_testing", END)
    return workflow.compile()


# One shared compiled graph coordinates all six StudyMate features.
study_graph = create_study_graph()


def run_study_agents(question, context="", topic="", task_type="general", uploaded_file=None):
    """Run the complete multi-agent workflow and return its final state."""
    return study_graph.invoke({
        "question": question,
        "topic": topic,
        "uploaded_file": uploaded_file,
        "context": context or "",
        "cleaned_context": "",
        "chunks": [],
        "prepared_context": "",
        "learner_profile": "",
        "answer": "",
        "task_type": task_type,
        "quality_check": "",
        "selected_agents": [],
        "quiz_history": list(st.session_state.get("quiz_history", [])),
    })


def generate_feature_content(instructions, context, task_type="general"):
    """Send feature-generation requests through the LangGraph agent team."""
    topic = st.session_state.get("study_topic", "") or "the uploaded study material"
    result = run_study_agents(
        question=instructions,
        context=context,
        topic=topic,
        task_type=task_type,
    )
    return result["answer"]


def parse_json_response(raw_text):
    """Parse JSON even if the model wraps it in a Markdown code fence."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The AI response was not in the expected format. Please generate it again.")
    return json.loads(cleaned[start:end + 1])

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# SESSION STATE
# ==========================================

defaults = {
    "started": False,
    "page": "Home",
    "study_topic": "",
    "study_file": None,
    "quiz_data": None,
    "quiz_submitted": False,
    "quiz_score": None,
    "quiz_history": [],
    "flashcards_data": None,
    "completed_summaries": 0,
    "study_plan_created": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page):
    st.session_state.page = page


def back_home():
    st.session_state.page = "Dashboard"


# ==========================================
# COMPLETE DESIGN CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body {
    color: white;
}

.stApp {
    color: white;
    background-color: #080d27 !important;
    background-image:
        radial-gradient(ellipse at 10% 20%, #287fff 0%, transparent 42%),
        radial-gradient(ellipse at 90% 15%, #a64dff 0%, transparent 40%),
        radial-gradient(ellipse at 50% 100%, #ec75d4 0%, transparent 38%),
        linear-gradient(130deg, #07132e, #17113d, #080d27);
    background-size: 250% 250%;
    animation: colourJourney 16s ease-in-out infinite alternate;
}

@keyframes colourJourney {
    0% { background-position: 0% 0%; }
    25% { background-position: 100% 20%; }
    50% { background-position: 70% 100%; }
    75% { background-position: 0% 70%; }
    100% { background-position: 100% 0%; }
}

/* Hide sidebar */
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background: transparent !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* Main text */
h1, h2, h3, h4 {
    color: #ffffff !important;
}

p, label {
    color: #e7e8ff;
}

/* Brand */
.brand {
    text-align: center;
    font-size: 29px;
    font-weight: 900;
    color: white;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.brand span {
    color: #d4aaff;
}

.tagline {
    text-align: center;
    color: #e0ddff;
    letter-spacing: 3px;
    font-size: 11px;
    font-weight: 700;
}

/* Hero */
.welcome {
    text-align: center;
    padding: 48px 30px 40px;
    border-radius: 30px;
    background: linear-gradient(
        120deg,
        rgba(63, 135, 255, 0.30),
        rgba(178, 80, 255, 0.28),
        rgba(241, 116, 204, 0.22)
    );
    border: 1px solid rgba(255,255,255,0.30);
    backdrop-filter: blur(25px);
    box-shadow: 0 20px 70px rgba(0,0,0,0.25);
    margin-bottom: 28px;
}

.welcome h1 {
    font-size: clamp(30px, 4vw, 48px);
    font-weight: 900;
    letter-spacing: -1.7px;
    margin: 18px 0;
}

.welcome p {
    font-size: 15px;
    line-height: 1.9;
    color: #ffffff !important;
    max-width: 720px;
    margin: auto;
}

.pill {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 30px;
    color: #ffffff;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.30);
    font-size: 12px;
    font-weight: 700;
}

/* Glass panels */
.panel {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.24);
    border-radius: 23px;
    padding: 25px;
    backdrop-filter: blur(22px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.17);
    margin-bottom: 18px;
}

.panel-title {
    font-size: 21px;
    font-weight: 800;
    color: white;
    margin-bottom: 9px;
}

.panel-subtitle {
    color: #e0e1ff;
    font-size: 13px;
    line-height: 1.7;
}

/* Feature cards */
div[data-testid="stButton"] button {
    width: 100%;
    min-height: 74px;
    border-radius: 18px;
    color: white !important;
    background: linear-gradient(
        130deg,
        rgba(76, 139, 255, 0.35),
        rgba(165, 85, 255, 0.32)
    ) !important;
    border: 1px solid rgba(255,255,255,0.30) !important;
    font-size: 16px;
    font-weight: 800;
    white-space: pre-wrap;
    transition: all 0.3s ease;
}

div[data-testid="stButton"] button:hover {
    transform: translateY(-5px);
    background: linear-gradient(
        130deg,
        rgba(75, 155, 255, 0.65),
        rgba(186, 85, 255, 0.60)
    ) !important;
    border-color: white !important;
    box-shadow: 0 12px 35px rgba(89, 100, 255, 0.30);
}

/* LIGHT INPUT BOXES */
.stTextInput input,
.stTextArea textarea {
    background: #f3f5ff !important;
    color: #151936 !important;
    border: 2px solid #a6baff !important;
    border-radius: 13px !important;
    font-size: 15px !important;
    box-shadow: 0 5px 20px rgba(0,0,0,0.12);
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #69718b !important;
    opacity: 1 !important;
}

.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stSlider label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Focus effect */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 2px solid #a66cff !important;
    box-shadow: 0 0 0 3px rgba(170,120,255,0.25) !important;
}

/* PDF UPLOAD BOX */
[data-testid="stFileUploader"] {
    background: #f1f4ff !important;
    border: 2px dashed #9674ff !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

[data-testid="stFileUploader"] * {
    color: #182044 !important;
}

[data-testid="stFileUploader"] button {
    background: #6651d9 !important;
    color: white !important;
    border: none !important;
}

/* DROPDOWN */
[data-baseweb="select"] > div {
    background: #f1f4ff !important;
    color: #151936 !important;
    border: 2px solid #a6a0ff !important;
    border-radius: 12px !important;
}

[data-baseweb="select"] input {
    color: #151936 !important;
}

[data-baseweb="select"] svg {
    fill: #30365a !important;
}

/* OUTPUT PANELS */
[data-testid="stAlert"] {
    background: rgba(12, 24, 65, 0.97) !important;
    border: 2px solid #9b9cff !important;
    border-radius: 16px !important;
    color: white !important;
    padding: 18px !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.18);
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #ffffff !important;
}

/* Custom AI answer panel */
.answer-box {
    background: linear-gradient(
        135deg,
        rgba(12, 25, 68, 0.98),
        rgba(49, 25, 91, 0.98)
    );
    border: 1px solid #9c9aff;
    border-radius: 18px;
    padding: 25px;
    color: white;
    margin-top: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.2);
}

.answer-box h3 {
    color: #d9c5ff !important;
}

.answer-box p {
    color: #ffffff !important;
    line-height: 1.8;
}

/* Radio and checkbox */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: white !important;
}

/* Buttons */
.stButton > button:focus {
    color: white !important;
    border-color: #d0b5ff !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #dedcff;
    font-size: 12px;
    padding: 25px 0;
}

hr {
    border-color: rgba(255,255,255,0.20) !important;
}

/* Mobile */
@media(max-width: 700px) {
    .welcome {
        padding: 30px 18px;
    }

    .welcome h1 {
        font-size: 29px;
    }

    .panel {
        padding: 18px;
    }
}

/* =====================================
   FINAL READABILITY FIX
===================================== */

/* Text input and text area */
.stTextInput input,
.stTextArea textarea {
    background-color: #F5F7FF !important;
    color: #171A38 !important;
    -webkit-text-fill-color: #171A38 !important;
    border: 2px solid #9B9EFF !important;
    border-radius: 14px !important;
    font-size: 15px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #626985 !important;
    -webkit-text-fill-color: #626985 !important;
    opacity: 1 !important;
}

/* Input labels */
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stSlider label,
.stFileUploader label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* PDF upload area */
[data-testid="stFileUploader"] {
    background: #F5F7FF !important;
    border: 2px dashed #9276FF !important;
    border-radius: 18px !important;
    padding: 18px !important;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #202449 !important;
}

/* Dropdown */
[data-baseweb="select"] > div {
    background: #F5F7FF !important;
    color: #171A38 !important;
}

[data-baseweb="select"] input {
    color: #171A38 !important;
    -webkit-text-fill-color: #171A38 !important;
}

/* Dropdown options */
[role="listbox"] {
    background: #F5F7FF !important;
}

[role="option"] {
    color: #171A38 !important;
}

/* AI answers and messages */
[data-testid="stAlert"] {
    background: #111D49 !important;
    border: 2px solid #A69BFF !important;
    border-radius: 16px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #FFFFFF !important;
}

/* Normal answer text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #FFFFFF !important;
    line-height: 1.8 !important;
}

/* Answer headings */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #E3D6FF !important;
}

/* Output panel */
.answer-box {
    background: #101B46 !important;
    border: 2px solid #A69BFF !important;
    border-radius: 18px !important;
    padding: 24px !important;
}

.answer-box p {
    color: #FFFFFF !important;
}

/* Buttons */
.stButton button {
    color: #FFFFFF !important;
}
/* ---------- INPUT FIELDS: LIGHT BACKGROUND + DARK TEXT ---------- */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
    background-color: #ffffff !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}

/* Dropdowns */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #172033 !important;
}

[data-testid="stSelectbox"] *,
[data-testid="stMultiSelect"] * {
    color: #172033 !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    border: 1px dashed #94a3b8 !important;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section * {
    color: #172033 !important;
}

/* ---------- OUTPUT: DARK PANEL + LIGHT TEXT ---------- */

.answer-box {
    background: rgba(15, 23, 42, 0.94) !important;
    color: #ffffff !important;
    padding: 22px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
}

.answer-box *,
.answer-box p,
.answer-box li,
.answer-box h1,
.answer-box h2,
.answer-box h3,
.answer-box strong {
    color: #ffffff !important;
}

/* General text outside input fields */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #ffffff;
}

/* Labels above input fields */
[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Alerts and status messages */
[data-testid="stAlert"] {
    background-color: #172033 !important;
    color: #ffffff !important;
}

[data-testid="stAlert"] * {
    color: #ffffff !important;
}



/* USER-PROVEN READABILITY FIXES — keep these LAST so they override earlier styles */
/* FIX UPLOADED FILE NAME AND STATUS */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section * {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

[data-testid="stFileUploader"] button {
    background: #6651d9 !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* FIX UPLOADED FILE STATUS BOX */
[data-testid="stAlert"] {
    background: #ffffff !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

/* FIX ALL INPUT TEXT */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background: white !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    caret-color: #172033 !important;
}

/* FIX DROPDOWN TEXT */
[data-testid="stSelectbox"] *,
[data-testid="stMultiSelect"] * {
    color: #172033 !important;
}

/* FIX AI OUTPUT */
.answer-box,
.answer-box *,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: white !important;
}

/* Keep generated AI markdown readable on the dark gradient */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="brand">🎓 StudyMate <span>AI</span></div>
<div class="tagline">YOUR INTELLIGENT LEARNING COMPANION</div>
""", unsafe_allow_html=True)

st.write("")


# ==========================================
# FIRST SCREEN
# ==========================================

if not st.session_state.started:

    st.markdown("""
    <div class="welcome">
        <div class="pill">✨ YOUR PERSONAL AI STUDY SPACE</div>
        <h1>Learn Smarter. Achieve More.</h1>
        <p>
        Your notes, your goals, your personal AI learning assistant.
        Upload your study material or tell us what you want to learn.
        Let's make your learning journey easier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">📚 Let's get started!</div>
        <div class="panel-subtitle">
        Upload your notes, enter a topic, or do both.
        Your StudyMate journey begins here.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="panel">
            <div style="font-size:35px">📄</div>
            <div class="panel-title">Upload your PDF</div>
            <div class="panel-subtitle">
            Upload lecture notes, textbooks or study material.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_pdf = st.file_uploader(
            "Choose your study PDF",
            type=["pdf"],
            key="initial_pdf"
        )

        if uploaded_pdf:
            st.success("PDF selected: " + uploaded_pdf.name)

    with right:
        st.markdown("""
        <div class="panel">
            <div style="font-size:35px">💡</div>
            <div class="panel-title">What do you want to learn?</div>
            <div class="panel-subtitle">
            Enter any subject or topic you want to understand.
            </div>
        </div>
        """, unsafe_allow_html=True)

        topic_input = st.text_area(
            "Your learning topic",
            placeholder="Example: DBMS, Python, Artificial Intelligence...",
            height=120,
            key="initial_topic"
        )

    st.write("")

    if st.button("🚀 Start My Learning Journey", type="primary"):
        if uploaded_pdf is None and not topic_input.strip():
            st.warning("Please upload a PDF or enter a topic.")
        else:
            st.session_state.study_file = uploaded_pdf
            st.session_state.study_topic = topic_input.strip()
            st.session_state.started = True
            st.session_state.page = "Dashboard"
            st.rerun()

    st.markdown("""
    <div class="footer">
    ✨ Learn • Practise • Improve • Grow
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# DASHBOARD
# ==========================================

elif st.session_state.page == "Dashboard":

    st.markdown("""
    <div class="welcome">
        <div class="pill">💜 WELCOME TO STUDYMATE AI</div>
        <h1>What shall we learn today?</h1>
        <p>
        Your personal learning space is ready.
        Choose a tool below and let's make learning easier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-title">📚 Your Study Material</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:
        if st.session_state.study_file:
            st.success("PDF: " + st.session_state.study_file.name)
        else:
            st.info("No PDF uploaded.")

    with b:
        if st.session_state.study_topic:
            st.success("Topic: " + st.session_state.study_topic)
        else:
            st.info("No topic entered.")

    if st.button("➕ Change PDF or Learning Topic"):
        st.session_state.started = False
        st.session_state.page = "Home"
        st.rerun()

    st.write("")

    st.markdown(
        '<div class="panel-title">Explore Your Study Tools ✨</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-subtitle">Click any feature to open it.</div>',
        unsafe_allow_html=True
    )

    features = [
        ("💬 Ask AI", "Ask AI"),
        ("📝 Smart Summary", "Smart Summary"),
        ("🧠 Smart Quiz", "Smart Quiz"),
        ("📚 Flashcards", "Flashcards"),
        ("📅 Study Planner", "Study Planner"),
        ("📊 My Progress", "My Progress")
    ]

    descriptions = {
        "Ask AI": "Get clear explanations for difficult topics.",
        "Smart Summary": "Convert notes into quick revision material.",
        "Smart Quiz": "Challenge yourself with practice questions.",
        "Flashcards": "Remember important concepts.",
        "Study Planner": "Organise your learning schedule.",
        "My Progress": "Monitor your learning journey."
    }

    for start in range(0, 6, 3):
        cols = st.columns(3, gap="large")

        for col, (title, page_name) in zip(
            cols, features[start:start + 3]
        ):
            with col:
                if st.button(
                    title,
                    key="card_" + page_name,
                    use_container_width=True
                ):
                    go_to(page_name)
                    st.rerun()

                st.markdown(
                    f'<div class="panel-subtitle">{descriptions[page_name]}</div>',
                    unsafe_allow_html=True
                )

    st.markdown(
        '<div class="footer">Made with 💜 for smarter learning</div>',
        unsafe_allow_html=True
    )


# ==========================================
# FEATURE PAGES
# ==========================================

else:

    if st.button("← Back to Dashboard"):
        back_home()
        st.rerun()

    st.write("")

    # --------------------------------------
    # ASK AI
    # --------------------------------------

  
    if st.session_state.page == "Ask AI":

        st.title("💬 Ask your StudyMate")
        st.write(
            "Ask questions about your uploaded notes or learning topic."
        )

        if st.session_state.study_file:
            st.success(
                "📄 Your PDF: " +
                st.session_state.study_file.name
            )

        question = st.text_area(
            "What would you like to understand?",
            placeholder="Example: Explain normalization in DBMS...",
            height=140,
            key="ask_ai_question"
        )

        if st.button("✨ Get My Answer"):

            if not question.strip():
                st.warning("Please enter your question.")

            else:
                try:
                    with st.spinner(
                        "⚡ Generating your answer..."
                    ):

                        result = run_study_agents(
                            question=question,
                            context="",
                            topic=st.session_state.study_topic,
                            task_type="ask_ai",
                            uploaded_file=st.session_state.study_file,
                        )
                        context = result.get("prepared_context", "")
                        answer = result["answer"]

                    st.markdown("""
                    <div class="answer-box">
                        <h3>✨ StudyMate AI Answer</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(answer)

                    if context:
                        with st.expander("📖 View information retrieved from your PDF"):
                            st.write(context)

                except Exception as error:
                    st.error("StudyMate AI could not get a response from Gemini.")
                    st.warning(f"Technical detail: {error}")


    # --------------------------------------
    # SMART SUMMARY
    # --------------------------------------

    elif st.session_state.page == "Smart Summary":

        st.title("📝 Smart Summary")
        st.write("Transform your study material into clear revision notes.")
        style = st.selectbox("Choose your summary style", ["Quick Revision", "Detailed Notes", "Exam Preparation", "Important Points"])

        if st.session_state.study_file:
            st.success("Selected PDF: " + st.session_state.study_file.name)
        elif st.session_state.study_topic:
            st.info("Topic: " + st.session_state.study_topic)

        if st.button("✨ Generate Summary"):
            if not st.session_state.study_file and not st.session_state.study_topic:
                st.warning("Please upload a PDF or enter a topic first.")
            else:
                try:
                    with st.spinner("⚡ Creating your summary..."):
                        context = get_study_context()
                        if st.session_state.study_file:
                            summary = generate_full_document_summary(style, context)
                        else:
                            prompt = f"Create a {style} study summary about {context}. Use clear headings and bullet points. Include key definitions, concepts, examples, and exam-important points. Explain the lesson fully in simple language and add clearly labelled extra explanation where useful. Return readable Markdown."
                            summary = generate_feature_content(prompt, context, task_type="summary")
                    st.markdown('<div class="answer-box"><h3>📝 Your Study Summary</h3></div>', unsafe_allow_html=True)
                    st.markdown(summary)
                    st.session_state.completed_summaries += 1
                except Exception as error:
                    st.error("I couldn't generate the summary.")
                    st.warning(f"Technical detail: {error}")

    # --------------------------------------
    # SMART QUIZ
    # --------------------------------------

    elif st.session_state.page == "Smart Quiz":

        st.title("🧠 Smart Quiz")
        st.write("Generate a practice quiz and check your score.")
        quiz_topic = st.text_input("Quiz topic", value=st.session_state.study_topic, key="quiz_topic_input")
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="quiz_difficulty")
        count = st.slider("Number of questions", 5, 20, 5, key="quiz_count")

        if st.button("🚀 Generate My Quiz"):
            if not quiz_topic.strip() and not st.session_state.study_file:
                st.warning("Please enter a topic or upload a PDF first.")
            else:
                try:
                    with st.spinner("⚡ Creating your quiz..."):
                        context = get_study_context() if st.session_state.study_file else quiz_topic
                        
                        prompt = f'Create exactly {count} {difficulty.lower()} multiple-choice questions about {quiz_topic or st.session_state.study_topic}. Use the supplied study material as the source when available. Return ONLY valid JSON in this shape: {{"questions":[{{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A. ...","explanation":"..."}}]}}. Each question needs four options and answer must exactly match one option. No Markdown fences or extra text.'

                        raw = generate_feature_content(prompt, context, task_type="quiz")
                        data = parse_json_response(raw)
                        questions = data.get("questions", [])
                        if not questions:
                            raise ValueError("The AI did not return any quiz questions.")
                        st.session_state.quiz_data = questions
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_score = None
                except Exception as error:
                    st.error("I couldn't create the quiz.")
                    st.warning(f"Technical detail: {error}")

        if st.session_state.quiz_data:
            st.markdown("### ✍️ Answer the questions")
            answers = []
            for i, item in enumerate(st.session_state.quiz_data):
                st.markdown(f"**Q{i + 1}. {item.get('question', 'Question')}**")
                options = item.get("options", [])
                selected = st.radio("Choose one answer", options, key=f"quiz_answer_{i}", index=None, label_visibility="collapsed")
                answers.append(selected)
            if st.button("✅ Submit Quiz", disabled=st.session_state.quiz_submitted):
                score = sum(1 for item, answer in zip(st.session_state.quiz_data, answers) if answer and answer.strip() == str(item.get("answer", "")).strip())
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True
                st.session_state.quiz_history.append({"score": score, "total": len(st.session_state.quiz_data), "topic": quiz_topic or st.session_state.study_topic})
                st.rerun()
            if st.session_state.quiz_submitted:
                total = len(st.session_state.quiz_data)
                score = st.session_state.quiz_score or 0
                st.success(f"Your score: {score} / {total} ({round(score / total * 100)}%)")
                for i, item in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**Q{i + 1}. Correct answer: {item.get('answer', 'Not provided')}**")
                    st.write(item.get("explanation", ""))
                if st.button("🔄 Create Another Quiz"):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.rerun()

    # --------------------------------------
    # FLASHCARDS
    # --------------------------------------

    elif st.session_state.page == "Flashcards":

        st.title("📚 Smart Flashcards")
        st.write("Turn important concepts into question-and-answer cards.")
        flash_topic = st.text_input("Enter a topic", value=st.session_state.study_topic, key="flash_topic_input")
        flash_count = st.slider("Number of flashcards", 5, 20, 8, key="flash_count")

        if st.button("✨ Create Flashcards"):
            if not flash_topic.strip() and not st.session_state.study_file:
                st.warning("Please enter a topic or upload a PDF first.")
            else:
                try:
                    with st.spinner("⚡ Creating your flashcards..."):
                        context = get_study_context() if st.session_state.study_file else flash_topic
                        if st.session_state.study_file:
                            cards = generate_full_document_flashcards(flash_topic or st.session_state.study_topic, flash_count, context)
                        else:
                            prompt = f'Create exactly {flash_count} useful study flashcards about {flash_topic or st.session_state.study_topic}. Return ONLY valid JSON: {{"flashcards":[{{"question":"...","answer":"..."}}]}}. Keep questions clear and answers concise. No Markdown fences or extra text.'
                            raw = generate_feature_content(prompt, context, task_type="flashcards")
                            data = parse_json_response(raw)
                            cards = data.get("flashcards", [])
                        if not cards:
                            raise ValueError("The AI did not return any flashcards.")
                        st.session_state.flashcards_data = cards
                except Exception as error:
                    st.error("I couldn't create flashcards.")
                    st.warning(f"Technical detail: {error}")

        if st.session_state.flashcards_data:
            st.markdown("### Your flashcards — hover over a card to flip it")

            # Render the cards in an isolated HTML component so Streamlit
            # displays the actual cards instead of showing the HTML source.
            card_html = []
            for i, card in enumerate(st.session_state.flashcards_data):
                question_text = html.escape(str(card.get("question", "Question"))).replace("\n", "<br>")
                answer_text = html.escape(str(card.get("answer", "Answer"))).replace("\n", "<br>")
                card_html.append(f"""
                <div class="sm-flash-card" tabindex="0" role="button" aria-label="Flashcard {i + 1}. Hover, tap, or focus to reveal answer">
                  <div class="sm-flash-inner">
                    <div class="sm-flash-face sm-flash-front">
                      <div class="sm-flash-label">QUESTION · CARD {i + 1}</div>
                      <div class="sm-flash-question">{question_text}</div>
                      <div class="sm-flash-hint">Hover or tap to reveal answer ↻</div>
                    </div>
                    <div class="sm-flash-face sm-flash-back">
                      <div class="sm-flash-label">ANSWER · CARD {i + 1}</div>
                      <div class="sm-flash-answer">{answer_text}</div>
                    </div>
                  </div>
                </div>
                """)

            cards_markup = "\n".join(card_html)
            rows_needed = (len(card_html) + 1) // 2
            component_height = min(2400, max(300, rows_needed * 270 + 50))
            flashcards_document = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <style>
                * {{ box-sizing: border-box; }}
                html, body {{ margin: 0; padding: 0; background: transparent; font-family: Arial, sans-serif; }}
                .sm-flash-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 8px 4px 24px; }}
                .sm-flash-card {{ background: transparent; min-height: 230px; height: 230px; perspective: 1100px; cursor: pointer; outline: none; }}
                .sm-flash-inner {{ position: relative; width: 100%; height: 100%; text-align: center; transition: transform .65s cubic-bezier(.2,.75,.25,1); transform-style: preserve-3d; }}
                .sm-flash-card:hover .sm-flash-inner, .sm-flash-card:focus .sm-flash-inner, .sm-flash-card.flipped .sm-flash-inner {{ transform: rotateY(180deg); }}
                .sm-flash-face {{ position: absolute; inset: 0; border-radius: 20px; padding: 24px 20px; display: flex; flex-direction: column; justify-content: center; align-items: center; backface-visibility: hidden; -webkit-backface-visibility: hidden; border: 1px solid rgba(190,180,255,.85); box-shadow: 0 12px 30px rgba(0,0,0,.22); color: #fff; overflow-y: auto; overflow-wrap: anywhere; }}
                .sm-flash-front {{ background: linear-gradient(135deg, #172b65, #56358b); }}
                .sm-flash-back {{ background: linear-gradient(135deg, #263b76, #6a3e91); transform: rotateY(180deg); }}
                .sm-flash-face, .sm-flash-face * {{ color: #fff !important; -webkit-text-fill-color: #fff !important; }}
                .sm-flash-label {{ font-size: 12px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; opacity: .85; margin-bottom: 14px; }}
                .sm-flash-question {{ font-size: 18px; font-weight: 750; line-height: 1.5; }}
                .sm-flash-answer {{ font-size: 16px; line-height: 1.6; }}
                .sm-flash-hint {{ font-size: 11px; margin-top: 18px; opacity: .85; }}
                @media (max-width: 600px) {{ .sm-flash-grid {{ grid-template-columns: 1fr; }} }}
                @media (hover: none) {{ .sm-flash-card:active .sm-flash-inner {{ transform: rotateY(180deg); }} }}
              </style>
            </head>
            <body>
              <div class="sm-flash-grid">{cards_markup}</div>
              <script>
                document.querySelectorAll('.sm-flash-card').forEach(card => {{
                  card.addEventListener('click', () => card.classList.toggle('flipped'));
                  card.addEventListener('keydown', event => {{
                    if (event.key === 'Enter' || event.key === ' ') {{
                      event.preventDefault(); card.classList.toggle('flipped');
                    }}
                  }});
                }});
              </script>
            </body>
            </html>"""
            st.components.v1.html(
                flashcards_document,
                height=component_height,
                scrolling=True
            )

    # --------------------------------------
    # STUDY PLANNER
    # --------------------------------------

    elif st.session_state.page == "Study Planner":

        st.title("📅 AI Study Planner")
        st.write("Create a realistic day-wise study schedule.")
        goal = st.text_input("What is your study goal?", value=st.session_state.study_topic, key="planner_goal")
        hours = st.slider("Daily study hours", 1, 12, 3, key="planner_hours")
        days = st.slider("Preparation days", 1, 30, 7, key="planner_days")

        if st.button("✨ Create My Study Plan"):
            if not goal.strip() and not st.session_state.study_file:
                st.warning("Please enter your learning goal or upload a PDF.")
            else:
                try:
                    with st.spinner("⚡ Building your study plan..."):
                        context = get_study_context() if st.session_state.study_file else goal
                        prompt = f"Create a practical {days}-day study timetable for this goal: {goal or st.session_state.study_topic}. The student can study {hours} hours per day. Divide each day into manageable sessions with topics, activities, short breaks and revision/practice. Use supplied material where possible. Format as a clear Markdown table with Day, Topics, Study activities and Time allocation. Finish with useful study tips."
                        plan = generate_feature_content(prompt, context, task_type="planner")
                    st.markdown('<div class="answer-box"><h3>📅 Your Personalised Study Plan</h3></div>', unsafe_allow_html=True)
                    st.markdown(plan)
                    st.session_state.study_plan_created = True
                except Exception as error:
                    st.error("I couldn't create your study plan.")
                    st.warning(f"Technical detail: {error}")

    # --------------------------------------
    # MY PROGRESS
    # --------------------------------------

    elif st.session_state.page == "My Progress":

        st.title("📊 My Learning Progress")
        st.markdown('<div class="welcome"><h1>Every step counts! 🌟</h1><p>Your learning activity and quiz performance are collected during this session.</p></div>', unsafe_allow_html=True)
        st.markdown("### 🤖 StudyMate Agent Team")
        st.caption("LangGraph coordinates these specialised stages when you use the study tools:")
        st.write("1. **Data Cleaning Agent** — extracts and cleans uploaded study notes.")
        st.write("2. **Data Training & Knowledge Index Agent** — fits a TF-IDF index to organise and retrieve relevant note sections. This does not fine-tune Gemini.")
        st.write("3. **Learner Evolution Agent** — reviews quiz history and identifies topics that may need more practice.")
        st.write("4. **Explanation Agent** — uses Gemini to create answers, summaries, quizzes, flashcards and study plans.")
        st.write("5. **Data Testing Agent** — checks generated responses and validates quiz/flashcard structure.")
        st.write("6. **LangGraph Orchestrator** — coordinates the agents in the workflow.")
        quiz_count_done = len(st.session_state.quiz_history)
        best_score = max([h["score"] / h["total"] * 100 for h in st.session_state.quiz_history], default=0)
        avg_score = (sum(h["score"] / h["total"] * 100 for h in st.session_state.quiz_history) / quiz_count_done) if quiz_count_done else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Quizzes completed", quiz_count_done)
        c2.metric("Best quiz score", f"{best_score:.0f}%")
        c3.metric("Average quiz score", f"{avg_score:.0f}%")
        st.markdown("### Your activity")
        st.write(f"📝 Summaries generated: **{st.session_state.completed_summaries}**")
        st.write(f"📅 Study plans created: **{1 if st.session_state.study_plan_created else 0}**")
        st.write(f"📚 Flashcards currently generated: **{len(st.session_state.flashcards_data or [])}**")
        if st.session_state.quiz_history:
            st.markdown("### Quiz history")
            for i, result in enumerate(reversed(st.session_state.quiz_history), 1):
                st.write(f"{i}. {result['topic']} — {result['score']}/{result['total']} ({round(result['score'] / result['total'] * 100)}%)")
        else:
            st.info("Complete a quiz to see your scores and learning progress here.")

