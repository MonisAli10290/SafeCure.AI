# import os
# import re
# import requests
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from threading import Lock

# from pypdf import PdfReader
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document


# app = Flask(__name__)
# CORS(app)
# rag_lock = Lock()
# VECTORSTORE = None


# # ==============================
# # LOAD PDFs
# # ==============================
# def load_pdfs(folder="data"):
#     docs = []
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#         return docs

#     for file in os.listdir(folder):
#         if file.endswith(".pdf"):
#             try:
#                 reader = PdfReader(os.path.join(folder, file))
#                 for page in reader.pages:
#                     text = page.extract_text()
#                     if text:
#                         docs.append(Document(page_content=text))
#             except Exception as e:
#                 print(f"Error loading {file}: {e}")
#     return docs

# # ==============================
# # INIT RAG
# # ==============================
# def initialize_rag():
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     if os.path.exists("faiss_index"):
#         return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#     docs = load_pdfs()
#     if not docs:
#         return FAISS.from_texts(["No guidelines loaded"], embeddings)

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(docs)

#     vs = FAISS.from_documents(chunks, embeddings)
#     vs.save_local("faiss_index")
#     return vs

# def get_vectorstore():
#     global VECTORSTORE
#     if VECTORSTORE is None:
#         with rag_lock:
#             if VECTORSTORE is None:
#                 VECTORSTORE = initialize_rag()
#     return VECTORSTORE

# # ==============================git rm --cached app.py

# # LLM CALL
# # ==============================
# def call_llm(llm_prompt):
    # import os
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#     try:
#         res = requests.post(
#             "https://api.groq.com/openai/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {GROQ_API_KEY}",
#                 "Content-Type": "application/json"
#             },
#             json={
#                 "model": "llama-3.3-70b-versatile",
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a strict Clinical Decision Support System. "
#                             "You MUST follow output format exactly. "
#                             "Never say 'analysis complete'. Always give diagnosis."
#                         )
#                     },
#                     {
#                         "role": "user",
#                         "content": llm_prompt
#                     }
#                 ],
#                 "max_tokens": 1000,
#                 "temperature": 0.2
#             },
#             timeout=30
#         )
#         print("STATUS:", res.status_code)
#         print("RAW:", res.text[:200])

#         if res.status_code == 200:
#             return res.json()["choices"][0]["message"]["content"]
#         else:
#             return f"⚠️ Error: {res.text[:200]}"

#     except Exception as e:
#         print("❌ ERROR:", e)
#         return f"⚠️ Error: {str(e)}"


# # ==============================
# # SAFETY FILTER
# # ==============================
# def safety_filter(response):
#     response = re.sub(r"\b\d+\s?(mg|g|ml|mcg|IU)\b", "[DOSAGE]", response, flags=re.IGNORECASE)

#     if "pregnan" in response.lower():
#         response = re.sub(r"\b(tetracycline|ciprofloxacin|ibuprofen|naproxen|aspirin)\b", 
#                       lambda m: f"⚠️ {m.group()} (AVOID in pregnancy)", 
#                       response, flags=re.IGNORECASE)
#     return response


# # ==============================
# # RESPONSE VALIDATION
# # ==============================
# def validate_response(response):
#     required_sections = [
#         "Clinical Assessment:",
#         "Antibiotic Necessity:",
#         "First-Line Therapy:",
#         "Second-Line Alternatives:",
#         "Contraindications & Precautions:"
#     ]

#     for section in required_sections:
#         if section not in response:
#             return False
#     return True


# # ==============================
# # CORE ENGINE
# # ==============================
# def clinical_engine(data):
#     vs = get_vectorstore()

#     query = f"""
# Symptoms: {data.get('condition')}
# Allergies: {data.get('allergies')}
# Medications: {data.get('medications')}
# """

#     try:
#         docs = vs.as_retriever(search_kwargs={"k": 3}).invoke(query)
#         context = "\n\n".join(d.page_content[:500] for d in docs)
#     except Exception:
#         context = "General medical guidelines"

#     llm_prompt = f"""
# PATIENT:
# Symptoms: {data.get('condition')}
# Allergies: {data.get('allergies')}
# Medications: {data.get('medications')}

# CLINICAL GUIDELINES:
# {context}

# You are a Clinical Decision Support System (CDSS).
# Your role is to perform clinical reasoning (not just give drugs) using safe, evidence-based logic.

# ========================
# CORE RULES (MANDATORY)
# ========================
# - Always give a MOST LIKELY diagnosis (no vague answers)
# - Never return:
#   • "Analysis complete"
#   • Empty sections
# - Always decide: Antibiotics → YES or NO
# - STRONGLY prefer NO antibiotic unless there is CLEAR bacterial evidence
# - Fever alone or Fever + joint pain = VIRAL until proven otherwise
# - If unsure → say "Most likely viral..." and recommend supportive care

# ========================
# ANTIBIOTIC GUARD (CRITICAL)
# ========================
# - Fever + joint pain ONLY → VIRAL (Dengue/Chikungunya/Viral Arthritis) → NO antibiotics
# - Fever + cold/cough/sore throat ONLY → Viral URI → NO antibiotics
# - Fever + rash + joint pain → Viral (Dengue, Chikungunya) → NO antibiotics
# - Septic Arthritis requires: fever + joint pain + swollen/hot/red joint → NOT just fever + joint pain
# - Do NOT diagnose Septic Arthritis without explicit signs: swollen joint, pus, inability to move joint
# - Burning urination + frequency → UTI → YES antibiotics
# - Productive cough + high fever + chest pain → Pneumonia → YES antibiotics
# - When in doubt → VIRAL → NO antibiotics → supportive care

# ========================
# CLINICAL LOGIC FLOW
# ========================
# 1. Identify syndrome from symptoms:
#    - Fever + joint pain ONLY → Viral (Dengue, Chikungunya, Viral Arthritis) → NO antibiotics
#    - Fever + cold/runny nose → Viral URI → NO antibiotics
#    - Burning urination → UTI → YES antibiotics
#    - Fever + joint swelling + redness + restricted movement → consider Septic Arthritis → YES antibiotics
#    - Fever + stiff neck + headache → consider Meningitis → YES antibiotics

# 2. Classify:
#    - Viral / Bacterial / Non-infectious
#    - Default to VIRAL if no strong bacterial indicator present

# 3. Antibiotic decision:
#    - Viral → supportive care: Paracetamol, hydration, rest
#    - Bacterial (with clear evidence) → YES antibiotics

# 4. Severity check:
#    - High fever + vomiting + inability to walk → escalate care
#    - Joint pain alone does NOT indicate Septic Arthritis

# 5. Safety check:
#    - Pregnancy → SAFE: Paracetamol, ORS, Vitamin C, Amoxicillin, Cephalosporins, Azithromycin
#    - Pregnancy → AVOID: Tetracyclines, Fluoroquinolones, Ibuprofen, Aspirin, Naproxen
#    - If pregnant + viral → recommend Paracetamol ONLY for pain/fever, NOT Ibuprofen
#    - Check allergies before any recommendation

# ========================
# TREATMENT RULES
# ========================
# IF antibiotics NOT needed (Viral):
# - Always recommend these specific medicines:
#   • Paracetamol (for fever and pain)
#   • ORS / Oral Rehydration Salts (for hydration)
#   • Vitamin C supplements (immune support)
#   • If joint pain severe → Ibuprofen (unless contraindicated)
# - Mention: Rest, fluid intake, avoid cold exposure
# - Do NOT leave First-Line Therapy as "Not required" for viral cases
# - Instead write specific supportive medicines under First-Line Therapy
# - Second-Line Alternatives for viral: "Not required"

# IF antibiotics ARE needed:
# - First-line: safest, narrow-spectrum
# - Second-line: alternative/stronger
# - Mention drugs to avoid (if any)

# ========================
# OUTPUT FORMAT (STRICT)
# ========================

# Clinical Assessment:
# <Most likely diagnosis + short reasoning>

# Antibiotic Necessity:
# YES / NO (with reason)

# First-Line Therapy:
# <list OR "Not required">

# Second-Line Alternatives:
# <list OR "Not required">

# Contraindications & Precautions:
# <unsafe drugs to avoid, special precautions, when to seek care>

# ========================
# SAFETY PRIORITY
# ========================
# - Avoid unnecessary antibiotics
# - Be conservative and clinically realistic
# - Always provide useful guidance (never empty)
# """

#     response = None
#     for i in range(3):
#         raw = call_llm(llm_prompt)
#         print("🔍 RAW:", raw)

#         if raw and validate_response(raw):
#             response = raw
#             break

#         print(f"⚠️ Retry {i+1}")

#     if not response:
#         response = """Clinical Assessment:
# Unable to determine. Please consult a doctor.

# Antibiotic Necessity:
# NO

# First-Line Therapy:
# Paracetamol, Hydration

# Second-Line Alternatives:
# Not required

# Contraindications & Precautions:
# Seek medical evaluation if symptoms persist
# """

#     response = safety_filter(response)
#     return response


# # ==============================
# # API ENDPOINTS
# # ==============================
# @app.route("/analyze", methods=["POST"])
# def analyze():
#     try:
#         data = request.get_json(force=True, silent=True)

#         if not data:
#             return jsonify({
#                 "status": "error",
#                 "response": "Invalid or missing JSON. Please send proper JSON with Content-Type: application/json"
#             }), 400

#         result = clinical_engine(data)
#         return jsonify({
#             "status": "success",
#             "response": result
#         })
#     except Exception as e:
#         return jsonify({"status": "error", "response": str(e)})


# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy"})


# @app.route("/")
# def home():
#     return render_template("index.html")


# # ==============================
# # RUN
# # ==============================
# if __name__ == "__main__":
#     print("🚀 Initializing Safecure AI...")
#     get_vectorstore()
#     print("✅ Ready! Visit http://localhost:5000")
#     app.run(debug=True, host='0.0.0.0', port=5000)
