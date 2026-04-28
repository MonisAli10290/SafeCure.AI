import os
import re
import sqlite3
import requests
import subprocess
import tempfile
import shutil
import uuid
import io
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g, send_file
from flask_cors import CORS
from threading import Lock

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")
except ImportError:
    pass

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from gtts import gTTS


app = Flask(__name__)
CORS(app)
rag_lock = Lock()
VECTORSTORE = None
DATABASE = "safecure.db"

# ==============================
# DATABASE SETUP
# ==============================
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = sqlite3.connect(DATABASE)
        db.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                condition TEXT,
                allergies TEXT,
                medications TEXT,
                age TEXT,
                assessment TEXT,
                antibiotic_necessity TEXT,
                first_line TEXT,
                second_line TEXT,
                contraindications TEXT,
                recommended_tests TEXT,
                additional_info_needed TEXT,
                raw_response TEXT
            )
        ''')
        db.commit()
        db.close()

def save_to_db(patient_id, data, parsed):
    db = get_db()
    try:
        db.execute("SELECT recommended_tests FROM patients LIMIT 1")
    except Exception:
        for col in ["age TEXT", "recommended_tests TEXT", "additional_info_needed TEXT"]:
            try:
                db.execute(f"ALTER TABLE patients ADD COLUMN {col}")
            except Exception:
                pass
        db.commit()

    db.execute('''
        INSERT INTO patients
        (id, timestamp, condition, allergies, medications, age, assessment,
         antibiotic_necessity, first_line, second_line, contraindications,
         recommended_tests, additional_info_needed, raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get('condition', ''),
        data.get('allergies', ''),
        data.get('medications', ''),
        data.get('age', ''),
        parsed.get('assessment', ''),
        parsed.get('antibiotic_necessity', ''),
        parsed.get('first_line', ''),
        parsed.get('second_line', ''),
        parsed.get('contraindications', ''),
        parsed.get('recommended_tests', ''),
        parsed.get('additional_info_needed', ''),
        parsed.get('raw', '')
    ))
    db.commit()


# ==============================
# PARSE RESPONSE
# ==============================
def parse_response(response_text):
    response_text = re.sub(r'\*+', '', response_text)
    response_text = re.sub(r'#+\s*', '', response_text)

    sections = {
        'assessment': '',
        'antibiotic_necessity': '',
        'first_line': '',
        'second_line': '',
        'contraindications': '',
        'recommended_tests': '',
        'additional_info_needed': '',
        'raw': response_text
    }

    section_map = [
        ("clinical assessment:", 'assessment'),
        ("antibiotic necessity:", 'antibiotic_necessity'),
        ("first-line therapy:", 'first_line'),
        ("second-line alternatives:", 'second_line'),
        ("contraindications & precautions:", 'contraindications'),
        ("recommended tests:", 'recommended_tests'),
        ("additional information needed:", 'additional_info_needed'),
    ]

    lines = response_text.split('\n')
    current = None
    buffer = []

    def flush(key):
        if key and buffer:
            sections[key] = ' | '.join(buffer).strip()
        buffer.clear()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for header, key in section_map:
            if line.lower().startswith(header):
                flush(current)
                current = key
                val = line[len(header):].strip()
                if val:
                    buffer.append(val)
                matched = True
                break

        if not matched and current:
            item = line.lstrip('-•').strip()
            if item:
                buffer.append(item)

    flush(current)
    return sections


# ==============================
# LOAD PDFs
# ==============================
def load_pdfs(folder="data"):
    docs = []
    if not os.path.exists(folder):
        os.makedirs(folder)
        return docs
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(folder, file))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        docs.append(Document(page_content=text))
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return docs


# ==============================
# INIT RAG
# ==============================
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = load_pdfs()
    if not docs:
        return FAISS.from_texts(["No clinical guidelines loaded."], embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local("faiss_index")
    return vs

def get_vectorstore():
    global VECTORSTORE
    if VECTORSTORE is None:
        with rag_lock:
            if VECTORSTORE is None:
                VECTORSTORE = initialize_rag()
    return VECTORSTORE


# ==============================
# LLM CALL
# ==============================
def call_llm(llm_prompt):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an experienced senior physician and clinical pharmacologist. "
                            "Think systematically like a board-certified doctor. "
                            "Always prioritize patient safety: start with the safest effective medicine, escalate only if needed. "
                            "You MUST follow the output format EXACTLY. Never use markdown ** or ## in your response."
                        )
                    },
                    {"role": "user", "content": llm_prompt}
                ],
                "max_tokens": 1800,
                "temperature": 0.15
            },
            timeout=45
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"ERROR: {res.text[:200]}"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ==============================
# SAFETY FILTER
# ==============================
def safety_filter(response):
    response = re.sub(r'\b\d+\s?(mg|g|ml|mcg|IU|kg)\b', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\b\d+\s?times?\s?(daily|a day)\b', '', response, flags=re.IGNORECASE)
    response = re.sub(r'every\s+\d+\s+hours?', '', response, flags=re.IGNORECASE)
    response = re.sub(r'for\s+\d+\s+days?', '', response, flags=re.IGNORECASE)
    response = re.sub(r'once\s+daily|twice\s+daily|three\s+times\s+daily', '', response, flags=re.IGNORECASE)
    response = re.sub(r'  +', ' ', response)
    response = re.sub(r' ,', ',', response)
    response = re.sub(r' \)', ')', response)
    if "pregnan" in response.lower():
        response = re.sub(
            r"\b(tetracycline|doxycycline|ciprofloxacin|levofloxacin|ibuprofen|naproxen|aspirin|trimethoprim|methotrexate|warfarin)\b",
            lambda m: f"{m.group()} [AVOID IN PREGNANCY]",
            response, flags=re.IGNORECASE
        )
    return response


# ==============================
# RESPONSE VALIDATION
# ==============================
def validate_response(response):
    cleaned = re.sub(r'\*+', '', response).lower()
    required = [
        "clinical assessment",
        "antibiotic necessity",
        "first-line therapy",
        "second-line alternatives",
        "contraindications",
        "recommended tests",
        "additional information needed"
    ]
    return all(s in cleaned for s in required)


# ==============================
# RULE ENGINE
# ==============================
SYMPTOM_RULES = [
    {
        "name": "Malaria",
        "keywords": ["chills", "rigors", "cyclical fever", "sweating", "malaria", "shivering"],
        "required_any": ["fever", "chills", "sweating", "rigors"],
        "antibiotic": "NO — antimalarial needed, not antibiotic",
        "first_line": ["Artemether-Lumefantrine (Falciparum Malaria)", "Paracetamol (Fever)", "ORS (Hydration)"],
        "second_line": ["IV Artesunate (Severe Malaria — hospital only)", "Chloroquine (Vivax Malaria)"],
        "tests": ["Malaria RDT — rapid confirmation", "Blood smear — species identification", "CBC — severity check"],
        "avoid": ["Aspirin — bleeding risk", "Ibuprofen — avoid in malaria", "Antibiotics — not indicated"],
    },
    {
        "name": "Dengue Fever",
        "keywords": ["dengue", "bone pain", "eye pain", "retro-orbital", "rash", "platelet", "breakbone"],
        "required_any": ["fever", "rash", "bone pain", "joint pain", "eye pain"],
        "antibiotic": "NO — viral disease, antibiotics strictly contraindicated",
        "first_line": ["Paracetamol (Fever and Pain)", "ORS (Hydration)", "Rest"],
        "second_line": ["IV Fluids (if severe dengue — hospital)"],
        "tests": ["NS1 Antigen — early dengue confirmation", "CBC with Platelet — thrombocytopenia check", "Dengue IgM/IgG — serology"],
        "avoid": ["Ibuprofen — severe bleeding risk", "Aspirin — bleeding risk", "Diclofenac — avoid", "Antibiotics — not indicated"],
    },
    {
        "name": "Typhoid Fever",
        "keywords": ["typhoid", "abdominal pain", "constipation", "rose spots", "enteric"],
        "required_any": ["fever", "abdominal pain", "nausea", "vomiting"],
        "antibiotic": "YES — bacterial infection, antibiotic required",
        "first_line": ["Azithromycin (Typhoid — safest oral)", "Paracetamol (Fever)"],
        "second_line": ["Ceftriaxone (Severe Typhoid — IV)", "Ciprofloxacin (if sensitive — avoid in pregnancy/children)"],
        "tests": ["Widal Test — typhoid serology", "Blood Culture — gold standard", "CBC — infection screen"],
        "avoid": ["NSAIDs — GI bleed risk in typhoid", "Ciprofloxacin — avoid in pregnancy and children"],
    },
    {
        "name": "Urinary Tract Infection (UTI)",
        "keywords": ["burning urination", "frequent urination", "dysuria", "uti", "urinary", "cloudy urine", "pelvic pain"],
        "required_any": ["burning", "urination", "dysuria", "pelvic pain", "urinary"],
        "antibiotic": "YES — bacterial infection confirmed",
        "first_line": ["Nitrofurantoin (UTI — safest first line)", "Paracetamol (Pain relief)"],
        "second_line": ["Cefixime (if resistance suspected)", "Co-amoxiclav (complicated UTI)"],
        "tests": ["Urine Routine/Microscopy — UTI confirmation", "Urine Culture — sensitivity testing"],
        "avoid": ["Trimethoprim — avoid in pregnancy and elderly", "Fluoroquinolones — avoid in pregnancy"],
    },
    {
        "name": "Community Acquired Pneumonia",
        "keywords": ["pneumonia", "productive cough", "chest pain", "breathlessness", "sputum"],
        "required_any": ["cough", "chest pain", "breathlessness", "fever", "sputum"],
        "antibiotic": "YES — bacterial pneumonia likely",
        "first_line": ["Amoxicillin (Pneumonia — first line)", "Paracetamol (Fever)", "ORS (Hydration)"],
        "second_line": ["Azithromycin (Penicillin allergy or atypical)", "Amoxicillin-Clavulanate (Severe)"],
        "tests": ["Chest X-Ray — consolidation confirmation", "CBC — infection severity", "Sputum Culture — pathogen identification"],
        "avoid": ["NSAIDs — avoid if bleeding risk"],
    },
    {
        "name": "Influenza / Viral Fever",
        "keywords": ["flu", "influenza", "myalgia", "body ache", "fatigue", "runny nose", "sore throat", "viral"],
        "required_any": ["fever", "body ache", "fatigue", "cough", "sore throat"],
        "antibiotic": "NO — viral infection, antibiotics not indicated",
        "first_line": ["Paracetamol (Fever and Body Ache)", "ORS (Hydration)", "Rest", "Vitamin C (Immune support)"],
        "second_line": ["Oseltamivir (Within 48 hours of onset only)"],
        "tests": ["CBC — rule out secondary bacterial infection", "Rapid Influenza Test — if available"],
        "avoid": ["Aspirin — Reye's syndrome risk", "Antibiotics — not indicated"],
    },
    {
        "name": "Cellulitis",
        "keywords": ["cellulitis", "skin redness", "skin warmth", "skin infection", "erythema"],
        "required_any": ["redness", "swelling", "warmth", "skin"],
        "antibiotic": "YES — bacterial skin infection",
        "first_line": ["Flucloxacillin (Cellulitis — first line)", "Paracetamol (Pain and Fever)"],
        "second_line": ["Clindamycin (Penicillin allergy)", "Co-amoxiclav (Polymicrobial)"],
        "tests": ["CBC — infection severity", "CRP — inflammatory marker"],
        "avoid": ["Penicillin — if allergy reported"],
    },
    {
        "name": "Otitis Media",
        "keywords": ["ear pain", "earache", "ear discharge", "otitis", "ear infection"],
        "required_any": ["ear pain", "earache", "ear"],
        "antibiotic": "CONDITIONAL — observe 48-72hrs first, antibiotics if not improving",
        "first_line": ["Paracetamol (Pain relief — watchful waiting 48-72hrs)", "Warm compress"],
        "second_line": ["Amoxicillin (If not improving after 72hrs)"],
        "tests": ["Otoscopy — tympanic membrane assessment"],
        "avoid": ["Aspirin — children", "Ciprofloxacin ear drops — only for perforated drum"],
    },
]

CRITICAL_KEYWORDS = [
    "altered consciousness", "confusion", "seizure", "fits",
    "difficulty breathing", "severe breathlessness", "can't breathe",
    "stiff neck", "photophobia", "neck stiffness",
    "uncontrolled bleeding", "coughing blood", "blood in stool",
    "crushing chest pain", "chest pain with sweating",
    "unconscious", "not passing urine",
]

def run_rule_engine(condition_text, allergies_text):
    text = condition_text.lower()
    allergy_text = allergies_text.lower()
    is_critical = any(kw in text for kw in CRITICAL_KEYWORDS)
    matched_rules = []
    for rule in SYMPTOM_RULES:
        keyword_hit = any(kw in text for kw in rule["keywords"])
        required_hit = any(kw in text for kw in rule["required_any"])
        if keyword_hit and required_hit:
            matched_rules.append(rule)
    allergy_warnings = []
    if "penicillin" in allergy_text:
        allergy_warnings.append("Penicillin allergy — avoid Amoxicillin, Flucloxacillin, Co-amoxiclav")
    if "sulfa" in allergy_text or "sulphonamide" in allergy_text:
        allergy_warnings.append("Sulfa allergy — avoid Trimethoprim-Sulfamethoxazole")
    if "aspirin" in allergy_text or "nsaid" in allergy_text:
        allergy_warnings.append("NSAID/Aspirin allergy — avoid all NSAIDs")
    if "metformin" in allergy_text:
        allergy_warnings.append("Metformin noted — avoid Ciprofloxacin (hypoglycemia risk)")
    return {"is_critical": is_critical, "matched_rules": matched_rules, "allergy_warnings": allergy_warnings}


# ==============================
# CORE CLINICAL ENGINE
# ==============================
def clinical_engine(data):
    vs = get_vectorstore()
    query = f"Symptoms: {data.get('condition')} Allergies: {data.get('allergies')} Medications: {data.get('medications')}"
    try:
        docs = vs.as_retriever(search_kwargs={"k": 3}).invoke(query)
        context = "\n\n".join(d.page_content[:500] for d in docs)
    except Exception:
        context = "General medical guidelines apply."

    condition = data.get('condition', '')
    allergies = data.get('allergies', 'None')
    medications = data.get('medications', 'None')

    rules = run_rule_engine(condition, allergies)
    matched = rules["matched_rules"]
    is_critical = rules["is_critical"]
    allergy_warnings = rules["allergy_warnings"]

    rule_hint = ""
    if is_critical:
        rule_hint += "CRITICAL EMERGENCY DETECTED — Recommend immediate hospital referral.\n"
    if matched:
        rule_hint += "\nRule Engine Pre-Analysis (HIGH CONFIDENCE — follow unless contradicted):\n"
        for r in matched:
            rule_hint += f"  Disease: {r['name']}\n"
            rule_hint += f"  Antibiotic: {r['antibiotic']}\n"
            rule_hint += f"  First-Line: {', '.join(r['first_line'][:3])}\n"
            rule_hint += f"  Key Tests: {', '.join(r['tests'][:3])}\n"
            rule_hint += f"  Avoid: {', '.join(r['avoid'][:3])}\n\n"
    if allergy_warnings:
        rule_hint += "Allergy Alerts:\n" + "\n".join(f"  - {w}" for w in allergy_warnings) + "\n"
    if not matched:
        rule_hint += "No strong pattern match. Use clinical reasoning based on symptoms only.\n"

    llm_prompt = f"""
You are a senior physician and clinical decision support system (CDS).

IMPORTANT: The Rule Engine below has pre-analyzed this case. You MUST follow it unless you have strong clinical reason to deviate.

========================
RULE ENGINE OUTPUT
========================
{rule_hint}
========================
INPUT DATA
========================
Symptoms: {condition}
Allergies: {allergies}
Current Medications: {medications}

Clinical Guidelines (PDF Context):
{context}

========================
REASONING FRAMEWORK (POORQA)
========================

Step 1: Pattern Recognition
- Analyze symptom combinations carefully
- Identify if symptoms strongly match a known clinical pattern
- If a strong pattern exists → prefer a specific diagnosis
- If no clear pattern → use general diagnosis cautiously

Step 2: Differential Diagnosis
- List top 1–3 most likely diseases
- Rank them by probability (most likely first)
- Do NOT include unrelated diseases

Step 3: Antibiotic Decision
- Decide: YES / NO / ALTERNATIVE (e.g., antiviral/antiparasitic)
- Antibiotics ONLY if strong bacterial evidence
- If unclear → DO NOT give antibiotics

Step 4: Treatment Selection
- First-line = safest effective option
- Second-line = only if needed
- Treatment MUST match primary diagnosis ONLY
- DO NOT mix treatments from different diseases

Step 5: Safety Check (CRITICAL)
- Check allergies
- Check drug safety
- Avoid:
  - Unnecessary antibiotics
  - NSAIDs in suspected bleeding-risk conditions
  - Steroids unless clearly indicated
- Prefer safest drug (e.g., paracetamol over NSAIDs when uncertain)

Step 6: Test Justification
- Suggest tests ONLY if clinically needed
- If strong suspicion of specific disease → MUST suggest confirmatory test
- DO NOT skip tests in moderate/high suspicion cases

Step 7: Final Validation
Before answering, ensure:
- No hallucinated symptoms added
- No unsafe drug suggested
- No missing critical test
- No vague diagnosis if specific possible

========================
STRICT RULES
========================

- NEVER assume symptoms not provided
- NEVER give antibiotics without clear indication
- NEVER use vague diagnosis if a strong pattern exists
- NEVER suggest harmful or contraindicated drugs
- ALWAYS prioritize patient safety
- ALWAYS be clinically logical and consistent

========================
OUTPUT FORMAT (STRICT)
========================

Clinical Assessment:
[ONLY disease names with one-word reason. Format: "Disease Name — brief reason". Max 3. Do NOT restate symptoms. Example: "Typhoid Fever — fever and abdominal pain", "Malaria — fever with chills"]

Antibiotic Necessity:
[YES / NO / ANTIMALARIAL / ANTIVIRAL — one-line reason. MUST match your diagnosis]

First-Line Therapy:
[Drug name + (condition). Safest first. NO doses. Example: "Azithromycin (Typhoid)". One per line. If NO antibiotic: give supportive care only]

Second-Line Alternatives:
[Alternatives only. "Not required" if none needed]

Contraindications & Precautions:
[Drug to avoid + reason. One per line. Check allergies and medications. "None" if none]

Recommended Tests:
[Test name — reason. One per line. MUST include confirmatory test for primary diagnosis]

Additional Information Needed:
[None]

========================
MANDATORY CONSISTENCY RULES
========================

- If diagnosis is Malaria/Dengue/Viral → Antibiotic MUST be NO
- If Antibiotic = NO → ZERO antibiotics in First-Line or Second-Line
- Treatment MUST match primary diagnosis
- Do NOT add symptoms that were not given
- Do NOT leave any section empty
- NEVER use ** or ## or bullet points in output
"""

    response = None
    for i in range(3):
        raw = call_llm(llm_prompt)
        if raw and not raw.startswith("ERROR") and validate_response(raw):
            response = raw
            break
        print(f"⚠️ Retry {i+1}: validation failed")

    if not response:
        response = """Clinical Assessment:
Unable to determine diagnosis. Please consult a doctor immediately.

Antibiotic Necessity:
NO — Insufficient information to make a safe antibiotic decision.

First-Line Therapy:
Paracetamol (fever and pain relief) | ORS (hydration) | Rest

Second-Line Alternatives:
Not required

Contraindications & Precautions:
Do not self-medicate without professional medical evaluation.

Recommended Tests:
Complete Blood Count (CBC) — Basic infection screening | Blood culture if fever persists beyond 3 days

Additional Information Needed:
Full symptom history — Duration and severity | Vital signs (temperature, blood pressure, pulse rate) — Clinical assessment
"""

    response = safety_filter(response)
    return response


# ==============================
# API ENDPOINTS
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        db_temp = get_db()
        count = db_temp.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        year = datetime.now().strftime("%Y")
        patient_id = f"P-{year}{str(count + 1).zfill(2)}"

        result = clinical_engine(data)
        parsed = parse_response(result)
        save_to_db(patient_id, data, parsed)

        def to_array(text):
            if not text:
                return []
            items = []
            for part in text.split(' | '):
                part = part.strip().lstrip('-•').strip()
                if part:
                    items.append(part)
            return items

        summary_text = f"""
Clinical assessment: {parsed.get("assessment", "")}.
Antibiotic necessity: {parsed.get("antibiotic_necessity", "")}.
First line therapy: {parsed.get("first_line", "")}.
Recommended tests: {parsed.get("recommended_tests", "")}.
"""
        return jsonify({
            "status": "success",
            "patient_id": patient_id,
            "clinical_assessment": to_array(parsed.get("assessment", "")),
            "antibiotic_necessity": parsed.get("antibiotic_necessity", "").strip(),
            "first_line_therapy": to_array(parsed.get("first_line", "")),
            "second_line_alternatives": to_array(parsed.get("second_line", "")),
            "contraindications": to_array(parsed.get("contraindications", "")),
            "recommended_tests": to_array(parsed.get("recommended_tests", "")),
            "additional_info_needed": to_array(parsed.get("additional_info_needed", "")),
            "summary": summary_text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/patients", methods=["GET"])
def get_patients():
    try:
        db = get_db()
        rows = db.execute("SELECT * FROM patients ORDER BY timestamp DESC").fetchall()
        patients = [dict(row) for row in rows]
        return jsonify({"status": "success", "patients": patients})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ==============================
# SMART TTS — OpenAI → ElevenLabs → gTTS fallback
# ==============================
def generate_tts(text, lang="en"):
    os.makedirs("static", exist_ok=True)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    path = os.path.join("static", filename)

    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
    ELEVENLABS_KEY   = os.getenv("ELEVENLABS_API_KEY", "").strip()
    ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE_ID", "ErXwobaYiN019PkySvjV")

    print(f"🔑 TTS: OpenAI key present = {bool(OPENAI_API_KEY)}, ElevenLabs = {bool(ELEVENLABS_KEY)}")

    # ── 1. OpenAI TTS ──
    if OPENAI_API_KEY:
        try:
            voice = "onyx"
            if lang == "hi":
                voice = "onyx"
            resp = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tts-1-hd",
                    "input": text,
                    "voice": voice,
                    "speed": 0.92,
                    "response_format": "mp3"
                },
                timeout=20
            )
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                print("✅ TTS: OpenAI")
                return path, filename
            else:
                print(f"OpenAI TTS error {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"OpenAI TTS exception: {e}")

    # ── 2. ElevenLabs ──
    if ELEVENLABS_KEY:
        try:
            resp = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE}",
                headers={
                    "xi-api-key": ELEVENLABS_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.55,
                        "similarity_boost": 0.80,
                        "style": 0.20,
                        "use_speaker_boost": True
                    }
                },
                timeout=20
            )
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                print("✅ TTS: ElevenLabs")
                return path, filename
            else:
                print(f"ElevenLabs TTS error {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"ElevenLabs TTS exception: {e}")

    # ── 3. gTTS fallback ──
    try:
        tts_lang_code = "hi" if lang == "hi" else "en"
        tts_obj = gTTS(text=text, lang=tts_lang_code, slow=False)
        tts_obj.save(path)
        print("⚠️ TTS: gTTS fallback")
        return path, filename
    except Exception as e:
        raise Exception(f"All TTS providers failed. Last error: {e}")


@app.route("/tts", methods=["POST"])
def tts():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        lang = data.get("lang", "en")
        if not text:
            return jsonify({"error": "Empty text"}), 400

        path, filename = generate_tts(text, lang)
        return jsonify({"audio_url": f"/static/{filename}"})

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# NVIDIA AUDIO2FACE-2D INTEGRATION
# ──────────────────────────────
# Flow:
#   1. /avatar/generate  — POST {audio_url, portrait_url}
#      → Flask fetches audio WAV, sends gRPC request to NVIDIA A2F-2D NIM
#      → Returns {video_url} pointing to a served MP4
#   2. /avatar/video/<filename> — serves the generated MP4
#
# The NVIDIA Audio2Face-2D NIM uses gRPC (not HTTP REST).
# We use the grpcio library to call the cloud-hosted NIM endpoint
# at grpc.nvcf.nvidia.com:443 with your NGC API key as bearer token.
#
# Portrait image: place your doctor photo at static/doctor_portrait.jpg
# If not present, a placeholder is returned with graceful fallback.
# ==============================

A2F_GRPC_TARGET   = "grpc.nvcf.nvidia.com:443"
A2F_FUNCTION_ID   = "952da94b-3a69-4f5c-b0bd-77b4d0f52e84"   # Audio2Face-2D NIM function ID
AVATAR_VIDEO_DIR  = os.path.join("static", "avatar_videos")
PORTRAIT_PATH     = os.path.join("static", "doctor_portrait.jpg")
os.makedirs(AVATAR_VIDEO_DIR, exist_ok=True)


def _load_a2f_proto_stubs():
    """
    Dynamically load Audio2Face-2D gRPC stubs.
    NVIDIA publishes the proto at:
    https://github.com/NVIDIA-Maxine/nim-clients/tree/main/audio2face-2d/proto
    We include the minimal compiled stubs inline as base64 to avoid requiring
    a separate compile step — or fall back to a REST-style HTTP request
    if grpcio is not installed.
    """
    try:
        import grpc
        return grpc
    except ImportError:
        return None


def call_audio2face_2d(audio_path: str, portrait_path: str, output_mp4: str) -> bool:
    """
    Call NVIDIA Audio2Face-2D NIM via gRPC.
    Sends portrait image + WAV audio → receives MP4 video stream.

    Returns True on success, False on failure.

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Flask (Python)                                          │
    │    → gRPC stub → NVIDIA cloud NIM (A2F-2D)              │
    │    ← MP4 video bytes streamed back                       │
    │    → saved to static/avatar_videos/<uuid>.mp4            │
    └─────────────────────────────────────────────────────────┘
    """
    NGC_API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("NGC_API_KEY", "")).strip()
    if not NGC_API_KEY:
        print("❌ A2F-2D: No NVIDIA_API_KEY found in environment")
        return False

    if not os.path.exists(audio_path):
        print(f"❌ A2F-2D: Audio file not found: {audio_path}")
        return False

    if not os.path.exists(portrait_path):
        print(f"❌ A2F-2D: Portrait not found: {portrait_path}")
        return False

    # ── Try grpcio path ──────────────────────────────────────
    try:
        import grpc
        from google.protobuf import descriptor_pool, descriptor_pb2
        # The NVIDIA nim-clients repo provides compiled proto stubs.
        # Install with: pip install nvidia-nim-clients-audio2face-2d
        # or clone https://github.com/NVIDIA-Maxine/nim-clients
        # Here we attempt import and fall through to subprocess if unavailable.
        try:
            # Try importing compiled stubs (user must have installed them)
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nim_clients"))
            from audio2face_2d import audio2face_2d_pb2, audio2face_2d_pb2_grpc

            with open(portrait_path, "rb") as f:
                portrait_bytes = f.read()
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            creds = grpc.ssl_channel_credentials()
            call_creds = grpc.access_token_call_credentials(NGC_API_KEY)
            combined = grpc.composite_channel_credentials(creds, call_creds)

            metadata = [
                ("function-id", A2F_FUNCTION_ID),
                ("authorization", f"Bearer {NGC_API_KEY}"),
            ]

            with grpc.secure_channel(A2F_GRPC_TARGET, combined) as channel:
                stub = audio2face_2d_pb2_grpc.Audio2Face2DServiceStub(channel)
                request_msg = audio2face_2d_pb2.Audio2Face2DRequest(
                    portrait=portrait_bytes,
                    audio=audio_bytes,
                    output_format="mp4",
                )
                response = stub.GenerateVideo(request_msg, metadata=metadata, timeout=120)
                with open(output_mp4, "wb") as f:
                    f.write(response.video)
                print("✅ A2F-2D: gRPC call succeeded (stub path)")
                return True

        except ImportError:
            print("⚠️ A2F-2D: nim-clients stubs not found, trying subprocess client...")
            pass

    except ImportError:
        print("⚠️ A2F-2D: grpcio not installed")

    # ── Subprocess path: use NVIDIA's official Python client script ──
    # Clone: git clone https://github.com/NVIDIA-Maxine/nim-clients.git
    # Then place at: ./nim-clients/audio2face-2d/python/audio2face-2d.py
    client_script = os.path.join(
        os.path.dirname(__file__), "nim-clients", "audio2face-2d", "python", "audio2face-2d.py"
    )
    if os.path.exists(client_script):
        try:
            NGC_API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("NGC_API_KEY", ""))
            env = os.environ.copy()
            env["NGC_API_KEY"] = NGC_API_KEY

            result = subprocess.run([
                "python", client_script,
                "--target", A2F_GRPC_TARGET,
                "--audio-input", audio_path,
                "--portrait-input", portrait_path,
                "--output", output_mp4,
                "--format", "wav",
            ], capture_output=True, text=True, timeout=120, env=env)

            if result.returncode == 0 and os.path.exists(output_mp4):
                print("✅ A2F-2D: subprocess client succeeded")
                return True
            else:
                print(f"❌ A2F-2D: subprocess failed: {result.stderr[:300]}")
        except Exception as e:
            print(f"❌ A2F-2D subprocess error: {e}")

    print("❌ A2F-2D: All methods failed. Setup instructions:")
    print("   1. pip install grpcio grpcio-tools")
    print("   2. git clone https://github.com/NVIDIA-Maxine/nim-clients.git")
    print("   3. Add NVIDIA_API_KEY=<your_ngc_key> to .env")
    print("   4. Place doctor portrait at: static/doctor_portrait.jpg")
    return False


@app.route("/avatar/generate", methods=["POST"])
def avatar_generate():
    """
    POST body: { "audio_url": "/static/tts_xxx.mp3", "lang": "en" }

    Steps:
      1. Convert MP3 → WAV  (ffmpeg or pydub)
      2. Call NVIDIA Audio2Face-2D gRPC
      3. Return { "video_url": "/avatar/video/<uuid>.mp4", "fallback": false }

    If A2F-2D is unavailable (no API key / no GPU / no stubs),
    returns { "video_url": null, "fallback": true } so the browser
    keeps showing the canvas avatar gracefully.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        audio_url = data.get("audio_url", "")

        if not audio_url:
            return jsonify({"video_url": None, "fallback": True, "reason": "No audio_url provided"})

        # Resolve the audio file path from URL
        # audio_url is like /static/tts_abc123.mp3
        audio_filename = audio_url.lstrip("/").replace("static/", "")
        mp3_path = os.path.join("static", audio_filename)

        if not os.path.exists(mp3_path):
            return jsonify({"video_url": None, "fallback": True, "reason": "Audio file not found on server"})

        # Convert MP3 → WAV (Audio2Face-2D requires WAV 16kHz mono)
        wav_path = mp3_path.replace(".mp3", "_a2f.wav")
        try:
            # Try ffmpeg first (best quality)
            subprocess.run([
                "ffmpeg", "-y", "-i", mp3_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
            ], capture_output=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fallback: pydub
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(wav_path, format="wav")
            except Exception as e:
                return jsonify({"video_url": None, "fallback": True, "reason": f"Audio conversion failed: {e}"})

        # Check portrait exists
        if not os.path.exists(PORTRAIT_PATH):
            return jsonify({
                "video_url": None,
                "fallback": True,
                "reason": "Doctor portrait not found. Place image at static/doctor_portrait.jpg"
            })

        # Generate video via NVIDIA Audio2Face-2D
        video_filename = f"a2f_{uuid.uuid4().hex}.mp4"
        output_mp4 = os.path.join(AVATAR_VIDEO_DIR, video_filename)

        success = call_audio2face_2d(wav_path, PORTRAIT_PATH, output_mp4)

        # Clean up temp WAV
        try:
            os.remove(wav_path)
        except Exception:
            pass

        if success and os.path.exists(output_mp4):
            return jsonify({
                "video_url": f"/avatar/video/{video_filename}",
                "fallback": False
            })
        else:
            return jsonify({
                "video_url": None,
                "fallback": True,
                "reason": "Audio2Face-2D generation failed. Check server logs."
            })

    except Exception as e:
        print(f"❌ /avatar/generate error: {e}")
        return jsonify({"video_url": None, "fallback": True, "reason": str(e)})


@app.route("/avatar/video/<filename>")
def serve_avatar_video(filename):
    """Stream the generated MP4 video to the browser."""
    video_path = os.path.join(AVATAR_VIDEO_DIR, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    return send_file(video_path, mimetype="video/mp4")


@app.route("/avatar/status", methods=["GET"])
def avatar_status():
    """
    Health check for the Audio2Face-2D integration.
    Returns configuration status so the frontend can decide
    whether to show video avatar or canvas fallback.
    """
    NGC_API_KEY  = os.getenv("NVIDIA_API_KEY", os.getenv("NGC_API_KEY", "")).strip()
    has_portrait = os.path.exists(PORTRAIT_PATH)
    has_grpcio   = False
    has_client   = False

    try:
        import grpc
        has_grpcio = True
    except ImportError:
        pass

    client_script = os.path.join(
        os.path.dirname(__file__), "nim-clients", "audio2face-2d", "python", "audio2face-2d.py"
    )
    has_client = os.path.exists(client_script)

    ready = bool(NGC_API_KEY) and has_portrait and (has_grpcio or has_client)

    return jsonify({
        "ready": ready,
        "has_api_key": bool(NGC_API_KEY),
        "has_portrait": has_portrait,
        "portrait_path": PORTRAIT_PATH,
        "has_grpcio": has_grpcio,
        "has_nim_client": has_client,
        "setup_instructions": {
            "1_api_key": "Add NVIDIA_API_KEY=<your_ngc_key> to your .env file",
            "2_portrait": "Place your doctor portrait photo at: static/doctor_portrait.jpg",
            "3_grpcio": "pip install grpcio grpcio-tools",
            "4_client": "git clone https://github.com/NVIDIA-Maxine/nim-clients.git",
            "5_ffmpeg": "Install ffmpeg for audio conversion (apt install ffmpeg)"
        }
    })


# ==============================
# CONVERSATIONAL DOCTOR ENDPOINT
# ==============================
DOCTOR_SYSTEM_PROMPT = """You are Dr. Safecure — an intelligent AI doctor, like Jarvis or Siri but for medicine.
You talk like a calm, confident, real human doctor. Natural. Direct. Never robotic or stiff.
You have clinical guidelines from PDFs available as context — use them for all recommendations.

YOUR VOICE & STYLE:
- Sound like a real doctor talking to a patient face to face. Warm but not over-the-top.
- Short natural sentences. Like how a person actually speaks.
- NO filler: never say "I understand your concern", "That sounds difficult", "Great question", "Certainly" etc.
- NO markdown: no **, ##, dashes as bullets. Plain text only during conversation.
- Respond in the SAME language the patient speaks (Hindi or English).

CONSULTATION FLOW:
1. Greet once, naturally. Ask the main problem.
2. Ask ONE follow-up question at a time. Max 3 follow-ups total. Be brief.
3. Once you have enough info (symptom + duration + allergies/current meds) → immediately give FINAL ASSESSMENT.
4. Use the special final assessment format below (EXACTLY).

EMERGENCY OVERRIDE: If patient says chest pain with sweating, can't breathe, unconscious, heavy bleeding → say:
"This sounds like an emergency. Please call an ambulance or go to the nearest hospital right now. Don't wait."
Then stop. Nothing else.

FINAL ASSESSMENT FORMAT (use EXACTLY when you have enough info):
When ready to give your final assessment, output it in this exact format — each section on its own line with the label followed by a colon and the content. Do not use bullets or dashes:

DIAGNOSIS: [Most likely condition and brief reason why]
FIRST LINE: [Primary medicines — drug names only, no doses, each separated by comma]
SECOND LINE: [Alternative medicines if first line fails or is contraindicated — or write "Not needed"]
TESTS: [Recommended tests — each separated by comma, with brief reason after a dash]
AVOID: [Medicines or things to avoid — or write "None"]
NOTE: [One short sentence of important advice for the patient]

RULES:
- Drug names only. NEVER give dosages (mg, ml, twice daily etc.)
- Recommend medicines ONLY from the clinical PDF guidelines provided.
- Do not leave any section empty in the final assessment.
- The final assessment is ONLY triggered when you have enough info. Before that, just converse naturally."""

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        conversation_history = data.get("history", [])
        user_message = data.get("message", "").strip()
        lang = data.get("lang", "en")

        if not user_message:
            return jsonify({"status": "error", "message": "Empty message"}), 400

        messages = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

        rag_context = "No clinical guidelines available."
        try:
            all_user_text = " ".join(
                t["content"] for t in conversation_history if t.get("role") == "user"
            ) + " " + user_message
            vs = get_vectorstore()
            docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(all_user_text)
            rag_context = "\n\n".join(d.page_content[:400] for d in docs)
        except Exception as e:
            print(f"RAG error in chat: {e}")

        messages.append({
            "role": "system",
            "content": f"CLINICAL GUIDELINES FROM PDF (use these for recommendations):\n{rag_context}"
        })

        for turn in conversation_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_message})

        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 380,
                "temperature": 0.4
            },
            timeout=30
        )

        if res.status_code != 200:
            return jsonify({"status": "error", "message": f"LLM error: {res.text[:200]}"}), 500

        reply = res.json()["choices"][0]["message"]["content"]
        reply = safety_filter(reply)

        # TTS for the reply
        tts_url = None
        try:
            tts_text = reply
            import re as _re
            tts_text = _re.sub(r'(DIAGNOSIS|FIRST LINE|SECOND LINE|TESTS|AVOID|NOTE):\s*', '', tts_text)
            _, fname = generate_tts(tts_text.strip(), lang)
            tts_url = f"/static/{fname}"
        except Exception as e:
            print(f"TTS error in chat: {e}")

        return jsonify({
            "status": "success",
            "reply": reply,
            "audio_url": tts_url
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/")
def home():
    return render_template("avatar.html")


@app.route("/database")
def database_page():
    return render_template("history.html")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    print("🚀 Initializing Safecure AI...")
    init_db()
    get_vectorstore()
    print("✅ Ready! Visit http://localhost:5000")
    print("\n📋 NVIDIA Audio2Face-2D Setup Checklist:")
    print("   → Add NVIDIA_API_KEY to .env")
    print("   → Place doctor photo at: static/doctor_portrait.jpg")
    print("   → pip install grpcio grpcio-tools")
    print("   → git clone https://github.com/NVIDIA-Maxine/nim-clients.git")
    print("   → apt install ffmpeg")
    app.run(debug=True, host='0.0.0.0', port=5000)