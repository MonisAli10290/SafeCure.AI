import os
import re
import sqlite3
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from threading import Lock

# OpenMed NER — symptom entity extraction
try:
    from transformers import pipeline
    openmed_ner = pipeline(
        "token-classification",
        model="OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M",
        aggregation_strategy="simple"
    )
    OPENMED_AVAILABLE = True
    print("✅ OpenMed NER loaded")
except Exception as e:
    openmed_ner = None
    OPENMED_AVAILABLE = False
    print(f"⚠️ OpenMed NER not available: {e}")

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")   # ← yeh ek word change karo
except ImportError:
    pass

from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from gtts import gTTS
import io
import uuid


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

    # Multiple header variations mapped to same section key
    section_map = [
        (["clinical assessment:", "assessment:"], 'assessment'),
        (["antibiotic necessity:", "antibiotic:"], 'antibiotic_necessity'),
        (["first-line therapy:", "first-line:", "first line therapy:", "first line:"], 'first_line'),
        (["second-line alternatives:", "second-line:", "second line alternatives:", "second line:"], 'second_line'),
        (["contraindications & precautions:", "contraindications:", "precautions:"], 'contraindications'),
        (["recommended tests:", "tests:", "investigations:"], 'recommended_tests'),
        (["additional information needed:", "additional information:", "additional info:"], 'additional_info_needed'),
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
        line_lower = line.lower()
        for headers, key in section_map:
            for header in headers:
                if line_lower.startswith(header):
                    flush(current)
                    current = key
                    val = line[len(header):].strip()
                    if val:
                        buffer.append(val)
                    matched = True
                    break
            if matched:
                break

        if not matched and current:
            item = line.lstrip('-•0123456789.').strip()
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
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                    "role": "system",
                    "content": (
                        "You are a senior physician and clinical decision support system. "
                        "Think like a board-certified doctor: diagnose precisely, treat conservatively, escalate when needed. "
                        "STRICT OUTPUT RULES: "
                        "Each section = maximum 2 lines. Do NOT repeat symptoms in assessment. "
                        "No full sentences — use structured labels only. No preamble. No conclusion. No filler text. "
                        "Do NOT use **, ##, or bullet dashes. Plain text only. "
                        "Antibiotics ONLY when bacterial infection is clearly and strongly indicated. "
                        "Patient safety is absolute — never suggest unsafe, contraindicated, or unnecessary drugs."
                    )
                },
                    {"role": "user", "content": llm_prompt}
                ],
                "max_tokens": 600,
                "temperature": 0.15
            },
            timeout=45
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"ERROR: {res.text[:200]}"
    except Exception as e:
        print(f"❌ GROQ EXCEPTION: {str(e)}")
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
    if not response or response.startswith("ERROR"):
        return False
    cleaned = re.sub(r'\*+', '', response).lower()

    # Each list = acceptable variations of same section header
    required_sections = [
        ["clinical assessment", "assessment:", "diagnosis:"],
        ["antibiotic necessity", "antibiotic:", "antibiotics:"],
        ["first-line therapy", "first-line", "first line", "first_line"],
        ["second-line", "second line", "second_line", "alternatives"],
        ["contraindication", "precaution", "avoid:"],
        ["recommended tests", "tests:", "investigations"],
        ["additional information", "additional info", "info needed"],
    ]

    matched = 0
    for section_variants in required_sections:
        if any(variant in cleaned for variant in section_variants):
            matched += 1

    # Pass if at least 4 out of 7 sections found (lenient — avoids false fallback)
    return matched >= 4


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
        "name": "Pyelonephritis (Upper UTI)",
        "keywords": ["flank pain", "loin pain", "back pain", "rigors", "vomiting", "high fever"],
        "required_any": ["burning urination", "dysuria", "urinary", "frequent urination"],
        "antibiotic": "YES — upper UTI requires systemic antibiotic",
        "first_line": ["Co-amoxiclav (Pyelonephritis — oral)", "Paracetamol (Fever)"],
        "second_line": ["Ceftriaxone IV (Severe — hospital only)", "Ciprofloxacin (if sensitivities allow — avoid in pregnancy)"],
        "tests": ["Urine Culture & Sensitivity — mandatory", "CBC — severity assessment", "Renal Function Tests — kidney involvement", "Ultrasound KUB — structural assessment"],
        "avoid": ["Nitrofurantoin — NOT effective for upper UTI", "Trimethoprim — resistance risk"],
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
    {
        "name": "Gastroenteritis / Food Poisoning",
        "keywords": ["diarrhea", "vomiting", "abdominal cramps", "loose stools", "food poisoning", "outside food", "gastroenteritis", "nausea"],
        "required_any": ["vomiting", "diarrhea", "abdominal cramps", "loose stools", "nausea"],
        "antibiotic": "NO — usually viral or self-limiting bacterial, antibiotics rarely needed",
        "first_line": ["ORS (Rehydration — priority)", "Paracetamol (Fever)", "Rest", "Zinc (if child)"],
        "second_line": ["Not applicable — supportive care sufficient in most cases"],
        "tests": ["Stool Routine — only if bloody diarrhea or fever > 3 days", "Stool Culture — if Salmonella/Shigella suspected"],
        "avoid": ["Antibiotics — not indicated for routine gastroenteritis", "Ibuprofen — GI irritation risk"],
    },
    {
        "name": "Acute Sinusitis (Viral)",
        "keywords": ["facial pain", "nasal congestion", "sinus", "sinusitis", "pressure around eyes", "blocked nose"],
        "required_any": ["facial pain", "nasal congestion", "headache", "blocked nose", "sinus"],
        "antibiotic": "NO — >90% viral, antibiotics not indicated in first 10 days",
        "first_line": ["Paracetamol (Pain and Fever)", "Saline nasal rinse (Congestion)", "Steam inhalation", "Rest"],
        "second_line": ["Amoxicillin (Only if symptoms worsen after 10 days or severe bacterial signs)"],
        "tests": ["None required — clinical diagnosis sufficient for mild sinusitis"],
        "avoid": ["Antibiotics in first 10 days — viral cause likely", "Decongestant sprays > 3 days — rebound congestion"],
    },
    {
        "name": "Tonsillitis / Pharyngitis",
        "keywords": ["sore throat", "throat pain", "difficulty swallowing", "tonsil", "pharyngitis", "strep"],
        "required_any": ["sore throat", "throat pain", "difficulty swallowing"],
        "antibiotic": "CONDITIONAL — viral in 70% of cases; antibiotic only if bacterial strep suspected",
        "first_line": ["Paracetamol (Pain and Fever)", "Salt water gargles", "Rest", "ORS (Hydration)"],
        "second_line": ["Amoxicillin (If Strep throat confirmed or strongly suspected — avoid if penicillin allergy)", "Azithromycin (Penicillin allergy)"],
        "tests": ["Throat Swab Culture — only if bacterial strep strongly suspected", "Rapid Strep Test — if available"],
        "avoid": ["Aspirin — children (Reye's syndrome)", "Amoxicillin — if penicillin allergy or EBV suspected (causes rash)"],
    },
    {
        "name": "Tuberculosis (TB)",
        "keywords": ["weight loss", "night sweats", "prolonged fever", "weeks", "tuberculosis", "tb", "haemoptysis", "blood in sputum"],
        "required_any": ["cough", "weight loss", "night sweats", "fever", "fatigue"],
        "antibiotic": "YES — anti-TB therapy required (specialist-supervised)",
        "first_line": ["Refer to specialist — TB requires DOTS therapy (specialist-supervised)", "Paracetamol (Fever)"],
        "second_line": ["DOTS Regimen (HRZE — Isoniazid, Rifampicin, Pyrazinamide, Ethambutol — specialist only)"],
        "tests": ["Chest X-Ray — bilateral infiltrates/cavitation", "Sputum AFB Smear — TB confirmation", "Mantoux Test — TB exposure", "CBNAAT/GeneXpert — rapid TB and drug resistance"],
        "avoid": ["Self-medication — incomplete treatment causes drug resistance", "Fluoroquinolones without TB ruled out — masks TB"],
    },
    {
        "name": "Sepsis / Septic Shock",
        "keywords": ["confusion", "low blood pressure", "hypotension", "fast breathing", "sepsis", "septic shock", "cold clammy", "organ failure"],
        "required_any": ["fever", "confusion", "low blood pressure", "fast breathing", "weakness"],
        "antibiotic": "YES — broad-spectrum IV antibiotics required immediately",
        "first_line": ["EMERGENCY: Immediate hospital referral required", "IV Piperacillin-Tazobactam (Broad-spectrum — hospital only)", "IV Normal Saline bolus (Fluid resuscitation)", "Norepinephrine (Vasopressor — ICU only if BP unresponsive)"],
        "second_line": ["IV Meropenem (If resistant organisms suspected)", "IV Vancomycin (If MRSA suspected)"],
        "tests": ["Blood Culture x2 — before antibiotics if possible", "CBC — infection severity", "Lactate — sepsis severity marker", "Renal Function Tests — organ involvement", "CRP/Procalcitonin — sepsis confirmation"],
        "avoid": ["Delay in antibiotics — every hour delay increases mortality", "Oral antibiotics — IV route required"],
    },
]

CRITICAL_KEYWORDS = [
    "altered consciousness", "confusion", "seizure", "fits",
    "difficulty breathing", "severe breathlessness", "can't breathe",
    "stiff neck", "photophobia", "neck stiffness",
    "uncontrolled bleeding", "coughing blood", "blood in stool",
    "crushing chest pain", "chest pain with sweating",
    "unconscious", "not passing urine",
    "high fever", "fever more than 5 days", "fever not responding to medication",
    "severe dehydration", "unable to swallow", "persistent vomiting",
    "severe headache with neck stiffness", "rash with fever",
    # Septic shock triggers
    "low blood pressure", "hypotension", "fast breathing", "rapid breathing",
    "septic shock", "sepsis", "organ failure", "cold clammy",
]

def run_rule_engine(condition_text, allergies_text, pregnancy_status="Not mentioned", diabetes="Not mentioned", age="Not specified"):
    text = condition_text.lower()
    allergy_text = allergies_text.lower()
    is_pregnant = pregnancy_status.lower() not in ["not mentioned", "no", "none", ""]
    is_diabetic = diabetes.lower() not in ["not mentioned", "no", "none", ""]
    is_child = any(w in text for w in ["child", "baby", "infant", "toddler", "kid"]) or \
               (age.isdigit() and int(age) < 18)

    is_critical = any(kw in text for kw in CRITICAL_KEYWORDS)
    matched_rules = []

    for rule in SYMPTOM_RULES:
        keyword_hit = any(kw in text for kw in rule["keywords"])
        required_hit = any(kw in text for kw in rule["required_any"])
        if keyword_hit and required_hit:
            matched_rules.append(rule)

    # PYELONEPHRITIS PRIORITY: fever + vomiting + back/flank pain → override plain UTI
    pyelonephritis_triggers = ["fever", "vomiting", "back pain", "flank pain", "loin pain", "rigors", "high fever"]
    has_upper_uti_flags = any(kw in text for kw in pyelonephritis_triggers)
    matched_names = [r["name"] for r in matched_rules]
    if "Pyelonephritis (Upper UTI)" in matched_names and "Urinary Tract Infection (UTI)" in matched_names and has_upper_uti_flags:
        matched_rules = [r for r in matched_rules if r["name"] != "Urinary Tract Infection (UTI)"]

    allergy_warnings = []
    if "penicillin" in allergy_text:
        allergy_warnings.append("Penicillin allergy — avoid Amoxicillin, Flucloxacillin, Co-amoxiclav")
    if "sulfa" in allergy_text or "sulphonamide" in allergy_text:
        allergy_warnings.append("Sulfa allergy — avoid Trimethoprim-Sulfamethoxazole")
    if "aspirin" in allergy_text or "nsaid" in allergy_text:
        allergy_warnings.append("NSAID/Aspirin allergy — avoid all NSAIDs")
    if "metformin" in allergy_text or is_diabetic:
        allergy_warnings.append("Diabetic patient — flag Ciprofloxacin interaction risk; prefer safer alternatives")
        allergy_warnings.append("Diabetic + UTI — Nitrofurantoin preferred over Fluoroquinolones")

    # Pregnancy-specific warnings — only if explicitly mentioned
    if is_pregnant:
        allergy_warnings.append("PREGNANCY NOTED — avoid Fluoroquinolones, Tetracyclines, Trimethoprim, NSAIDs, Clarithromycin")
        allergy_warnings.append("PREGNANCY + UTI — Nitrofurantoin safe only in 1st/2nd trimester; prefer Cefalexin if trimester unknown")
        allergy_warnings.append("PREGNANCY + Pneumonia — Azithromycin (Category B) preferred; Clarithromycin (Category C) — AVOID")

    # Child-specific warnings
    if is_child:
        allergy_warnings.append("CHILD PATIENT — avoid Aspirin (Reye's syndrome risk); avoid Fluoroquinolones under 18")
        allergy_warnings.append("CHILD PATIENT — weight-based dosing needed; flag in Additional Information Needed")

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
    age = data.get('age', 'Not specified')
    pregnancy_status = data.get('pregnancy', 'Not mentioned')
    diabetes = data.get('diabetes', 'Not mentioned')
    renal_issues = data.get('renal_issues', 'Not mentioned')

    # Run rule engine — pass all patient factors
    rules = run_rule_engine(condition, allergies, pregnancy_status, diabetes, age)
    matched = rules["matched_rules"]
    is_critical = rules["is_critical"]
    allergy_warnings = rules["allergy_warnings"]

    # Septic Shock override — if critical AND confusion + hypotension present → force emergency
    septic_flags = ["confusion", "low blood pressure", "hypotension", "fast breathing", "sepsis", "septic"]
    septic_hit = sum(1 for kw in septic_flags if kw in condition.lower())
    is_septic_shock = septic_hit >= 2

    # Build rule hints for LLM
    rule_hint = ""
    if is_critical:
        rule_hint += "CRITICAL EMERGENCY DETECTED — Recommend immediate hospital referral.\n"
    if is_septic_shock:
        rule_hint += (
            "SEPTIC SHOCK PATTERN DETECTED — MANDATORY RULES:\n"
            "  1. Clinical Assessment MUST start with: EMERGENCY: Immediate hospital referral required.\n"
            "  2. First-Line MUST include: IV Broad-Spectrum Antibiotics (Piperacillin-Tazobactam or Meropenem — hospital only)\n"
            "  3. First-Line MUST include: IV Fluid Resuscitation (Normal Saline bolus)\n"
            "  4. First-Line MUST include: Vasopressors if BP not responding (Norepinephrine — ICU only)\n"
            "  5. Additional Information Needed MUST say: EMERGENCY — ICU admission required immediately.\n"
        )
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
Age: {age}
Pregnancy Status: {pregnancy_status}
Diabetes: {diabetes}
Renal Issues: {renal_issues}

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
- Mild case + clear clinical pattern → 0 tests needed, write "None required"
- Moderate suspicion → 1–2 targeted confirmatory tests only
- Severe or uncertain diagnosis → targeted panel only, no blanket ordering
- DO NOT suggest CBC routinely for every case
- Each test MUST have a specific clinical reason stated after a dash

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
- NEVER give antibiotics without clear bacterial indication
- NEVER use vague diagnosis if a strong clinical pattern exists
- NEVER suggest contraindicated or harmful drugs
- ALWAYS prioritize patient safety above all else
- ALWAYS be clinically logical and consistent
- Viral fever / dengue / malaria / flu → antibiotics ABSOLUTELY PROHIBITED
- Mild URI or cold < 3 days → supportive care ONLY, no antibiotics
- If bacterial infection is UNCLEAR → do NOT give antibiotics; recommend 48hr monitoring
- If CRITICAL emergency (chest pain + sweating, unconscious, heavy bleeding, can't breathe, stiff neck) →
  Clinical Assessment MUST start with "EMERGENCY: Immediate hospital referral required."
- Consider patient age, allergies, comorbidities (diabetes, renal issues) in every recommendation
- Include pregnancy contraindications ONLY if pregnancy_status is explicitly mentioned by user — do NOT assume
- Elderly or renal impairment → avoid Nitrofurantoin for upper UTI; flag renally-cleared drug risks
- Diabetic patients → flag Ciprofloxacin interaction risk where relevant

========================
OUTPUT FORMAT (STRICT)
========================

Use EXACTLY these section headers in this order — do not rename, skip, or reorder:

Clinical Assessment:
[Top 2–3 differential diagnoses ranked by probability. Format each on its own line:
1. Most Likely: [Disease] — [one clinical reason from given symptoms only]
2. Also Consider: [Disease] — [one clinical reason]
3. Less Likely: [Disease] — [only if genuinely relevant]
Do NOT repeat symptoms. Do NOT invent symptoms not provided.]

Antibiotic Necessity:
[YES / NO / ANTIMALARIAL / ANTIVIRAL — one short reason. Must match primary diagnosis.]

First-Line Therapy:
[One drug per line. Format: DrugName (Condition). No doses.
If NO antibiotic: supportive care only — Paracetamol, ORS, Rest, Vitamin C as appropriate.
If YES antibiotic: safest guideline-based antibiotic first.]

Second-Line Alternatives:
[If antibiotic YES: 1–2 guideline-based alternatives, one per line.
If antibiotic NO: write — Not applicable — [reason e.g. viral infection]
NEVER write an antibiotic here when Antibiotic Necessity = NO.]

Contraindications & Precautions:
[Drug to avoid — reason. One per line.
Check allergies and current medications.
Pregnancy warnings ONLY if pregnancy was explicitly mentioned.
Write None if nothing applicable.]

Recommended Tests:
[TestName — specific reason. One per line.
Mild case with clear diagnosis → write: None required — clinical diagnosis sufficient
Moderate/severe or uncertain → targeted confirmatory tests only, no blanket panels.]

Additional Information Needed:
[EMERGENCY cases → first line MUST be: EMERGENCY: Immediate hospital referral required.
Clear mild diagnosis → write: None
Otherwise → max 2 missing details that would change management.]

========================
MANDATORY CONSISTENCY RULES
========================
- Malaria / Dengue / Viral Fever / Influenza / Common Cold → Antibiotic MUST be NO
- If Antibiotic = NO → zero antibiotics in First-Line AND Second-Line
- Treatment must match primary diagnosis only — no mixing
- Never add symptoms not given by user
- Never leave a section empty — use None or Not applicable
- Never use ** ## or bullet dashes — plain text only
- Follow exact section headers above

PYELONEPHRITIS vs UTI (CRITICAL):
- Fever + vomiting + back pain/flank pain + urinary symptoms → PRIMARY = Pyelonephritis
- Nitrofurantoin PROHIBITED in Pyelonephritis — ineffective in kidney tissue
- Pyelonephritis → Co-amoxiclav oral or Ceftriaxone IV (severe)
- Simple UTI (no fever, no systemic symptoms) → Nitrofurantoin acceptable

PREGNANCY SAFETY (only if explicitly mentioned):
- Nitrofurantoin → trimester unknown → prefer Cefalexin instead
- Azithromycin → Category B → safe, preferred for pneumonia
- Clarithromycin → Category C → AVOID, flag in contraindications
- Fluoroquinolones, Tetracyclines, Trimethoprim, NSAIDs → AVOID

PNEUMONIA ANTIBIOTIC DECISION:
- High fever + chest pain + breathlessness + productive cough → bacterial → antibiotic YES
- Fatigue + body ache + mild cough only → viral first → no antibiotic without X-Ray confirmation
- Always recommend Chest X-Ray as first test for suspected pneumonia

SEPTIC SHOCK (MANDATORY):
- Fever + confusion + low BP + fast breathing → Septic Shock
- Assessment MUST start: EMERGENCY: Immediate hospital referral required.
- First-Line MUST include IV antibiotics + IV fluids + vasopressors if BP unresponsive
- Additional Info MUST say: EMERGENCY — ICU admission required immediately

DIABETIC PATIENTS:
- Flag Ciprofloxacin interaction risk if patient is diabetic on relevant medications
- Prefer safer alternatives where possible

CHILD PATIENTS:
- Avoid Aspirin — Reye's syndrome risk
- Avoid Fluoroquinolones in children under 18
- Dose adjustments may be needed — flag this in Additional Information Needed
"""

    response = None
    last_raw = None
    for i in range(3):
        raw = call_llm(llm_prompt)
        if raw and not raw.startswith("ERROR"):
            last_raw = raw
            if validate_response(raw):
                response = raw
                break
            print(f"⚠️ Retry {i+1}: validation failed")
        else:
            import time
            time.sleep(10)  # 10 second wait before retry

    # Agar validation pass nahi hua lekin LLM ne kuch meaningful diya → use karo
    if not response:
        if last_raw and len(last_raw.strip()) > 100:
            print("⚠️ Using last LLM response despite validation failure")
            response = last_raw
        else:
            response = """Clinical Assessment:
Unable to determine diagnosis. Please consult a doctor immediately.

Antibiotic Necessity:
NO — Insufficient information to make a safe antibiotic decision.

First-Line Therapy:
Paracetamol (fever and pain relief) | ORS (hydration) | Rest

Second-Line Alternatives:
Not applicable — insufficient clinical information

Contraindications & Precautions:
Do not self-medicate without professional medical evaluation.

Recommended Tests:
Complete Blood Count (CBC) — Basic infection screening

Additional Information Needed:
Please provide full symptom history, duration, and severity to enable proper diagnosis.
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

        # ✅ FIX: Empty symptoms check — fallback se bachao
        condition = data.get('condition', '').strip()
        if not condition:
            return jsonify({
                "status": "error",
                "message": "Please enter symptoms before analyzing. The Symptoms / Condition field cannot be empty."
            }), 400
        result = clinical_engine(data)
        parsed = parse_response(result)


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
            "contraindications": [x for x in to_array(parsed.get("contraindications", "")) if x.lower() != "none"],
            "recommended_tests": to_array(parsed.get("recommended_tests", "")),
            "additional_info_needed": [x for x in to_array(parsed.get("additional_info_needed", "")) if x.lower() != "none"],
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
    """
    Tries TTS providers in order of quality:
    1. OpenAI TTS (onyx = deep calm male, nova = female)
    2. ElevenLabs
    3. gTTS (fallback)
    Returns: (filepath, filename) or raises Exception
    """
    os.makedirs("static", exist_ok=True)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    path = os.path.join("static", filename)

    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "").strip()
    ELEVENLABS_KEY   = os.getenv("ELEVENLABS_API_KEY", "").strip()
    ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE_ID", "ErXwobaYiN019PkySvjV")

    print(f"🔑 TTS: OpenAI key present = {bool(OPENAI_API_KEY)}, ElevenLabs = {bool(ELEVENLABS_KEY)}")

    # ── 1. OpenAI TTS (best quality, natural, no pauses) ──
    if OPENAI_API_KEY:
        try:
            voice = "onyx"   # Deep, calm male — Jarvis feel
            if lang == "hi":
                voice = "onyx"  # onyx handles Hindi well too
            resp = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tts-1-hd",      # HD = higher quality, smoother
                    "input": text,
                    "voice": voice,
                    "speed": 0.92,            # Calm, measured doctor pace
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

    # ── 2. ElevenLabs (most natural, best for Hindi too) ──
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

    # ── 3. gTTS fallback (basic but works) ──
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

ANTIBIOTIC RULES (NON-NEGOTIABLE):
- Viral fever, dengue, malaria, flu, cold, sore throat → NEVER prescribe antibiotics under any circumstance
- Antibiotics ONLY when bacterial infection is strongly and clearly suspected
- Mild URI / cough < 3 days → supportive care ONLY (Paracetamol, ORS, Rest)
- If bacterial vs viral is unclear → say "Monitor for 48 hours — antibiotics not needed yet"
- Include pregnancy contraindications ONLY if patient explicitly mentions being pregnant

RULES:

Step 1: Pattern Recognition
- Analyze symptom combinations carefully
- Identify if symptoms strongly match a known clinical pattern
- If a strong pattern exists → prefer a specific diagnosis
- If no clear pattern → use general diagnosis cautiously

Step 2: Differential Diagnosis
- List top 1 (or 2 if needed) most likely diseases
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
MANDATORY CONSISTENCY RULES
========================

- If diagnosis is Malaria/Dengue/Viral → Antibiotic MUST be NO
- If Antibiotic = NO → ZERO antibiotics in First-Line or Second-Line
- Treatment MUST match primary diagnosis
- Do NOT add symptoms that were not given
- Do NOT leave any section empty
- NEVER use ** or ## or bullet points in output"""

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

        # Build messages array for Groq
        messages = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

        # RAG context retrieve karo
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

        # RAG system context inject karo
        messages.append({
            "role": "system",
            "content": f"CLINICAL GUIDELINES FROM PDF (use these for recommendations):\n{rag_context}"
        })

        # Pehle conversation history add karo
        for turn in conversation_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        # OpenMed NER — entities extract karo user message se
        openmed_context = ""
        if OPENMED_AVAILABLE and openmed_ner:
            try:
                entities = openmed_ner(user_message)
                detected = list(set([
                    e['word'].strip() for e in entities
                    if e['score'] > 0.7 and len(e['word'].strip()) > 2
                ]))
                if detected:
                    openmed_context = f"\n[OpenMed detected clinical entities: {', '.join(detected)}]"
                    print(f"🔬 OpenMed entities: {detected}")
            except Exception as e:
                print(f"OpenMed NER error: {e}")

        # Enriched user message aakhir mein append karo
        enriched_message = user_message + openmed_context
        messages.append({"role": "user", "content": enriched_message})

        # Groq API call
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": 480,
                "temperature": 0.2,
            },
            timeout=30
        )

        if res.status_code != 200:
            return jsonify({"status": "error", "message": f"LLM error: {res.text[:200]}"}), 500

        reply = res.json()["choices"][0]["message"]["content"]
        reply = safety_filter(reply)

        # TTS generate karo
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
    return render_template("chatbot.html")


@app.route("/database")
def database_page():
    return render_template("history.html")


# ==============================
# RUN
# ==============================

if __name__ == '__main__':
    print("🚀 Initializing Safecure AI...")
    init_db()
    try:
        get_vectorstore()
    except Exception as e:
        print(f"⚠️ Vectorstore warning: {e}")
    # print("✅ Ready! Visit http://localhost:5000")
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)s
