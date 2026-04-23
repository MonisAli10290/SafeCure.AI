import os
import re
import sqlite3
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from threading import Lock

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
    # Migrate old schema if needed
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
    import os
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
    # Remove any dosage numbers that sneak through
    response = re.sub(r'\b\d+\s?(mg|g|ml|mcg|IU|kg)\b', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\b\d+\s?times?\s?(daily|a day)\b', '', response, flags=re.IGNORECASE)
    response = re.sub(r'every\s+\d+\s+hours?', '', response, flags=re.IGNORECASE)
    response = re.sub(r'for\s+\d+\s+days?', '', response, flags=re.IGNORECASE)
    response = re.sub(r'once\s+daily|twice\s+daily|three\s+times\s+daily', '', response, flags=re.IGNORECASE)
    # Clean up extra whitespace left behind
    response = re.sub(r'  +', ' ', response)
    response = re.sub(r' ,', ',', response)
    response = re.sub(r' \)', ')', response)
    # Pregnancy warnings
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

    # Run rule engine first
    rules = run_rule_engine(condition, allergies)
    matched = rules["matched_rules"]
    is_critical = rules["is_critical"]
    allergy_warnings = rules["allergy_warnings"]

    # Build rule hints for LLM
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/")
def home():
    return render_template("app.html")


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
    app.run(debug=True, host='0.0.0.0', port=5000)