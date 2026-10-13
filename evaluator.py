import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def sanitize_evaluation_data(data):
    """
    Enforces deterministic mathematical consistency on Gemini evaluation reports:
    - Clamps individual question marks between 0 and max_marks.
    - Synchronizes 3-part rubric breakdown (concept + diagram + working) to equal awarded_marks exactly.
    - Calculates total_awarded_marks and total_max_marks as the true mathematical sum of questions.
    - Recalculates aggregate percentage and VTU Grade Class deterministically.
    """
    if not isinstance(data, dict):
        return {
            "total_awarded_marks": 0.0,
            "total_max_marks": 10,
            "question_breakdown": [],
            "strengths": [],
            "missing_concepts": [],
            "general_feedback": "Invalid evaluation data."
        }

    questions = data.get("question_breakdown", [])
    if not isinstance(questions, list):
        questions = []
        data["question_breakdown"] = questions

    total_awarded = 0.0
    total_max = 0

    for idx, q in enumerate(questions):
        # 1. Sanitize max marks
        try:
            m_max = int(float(q.get("max_marks", 10)))
        except (ValueError, TypeError):
            m_max = 10
        if m_max <= 0:
            m_max = 10
        q["max_marks"] = m_max

        # 2. Sanitize awarded marks
        try:
            m_awarded = float(q.get("awarded_marks", 0.0))
        except (ValueError, TypeError):
            m_awarded = 0.0
        m_awarded = max(0.0, min(float(m_max), round(m_awarded, 1)))
        q["awarded_marks"] = m_awarded

        # 3. Synchronize 3-part rubric breakdown
        rb = q.get("rubric_breakdown")
        if not isinstance(rb, dict):
            c_max = max(1, round(m_max * 0.4))
            d_max = max(1, round(m_max * 0.3))
            e_max = max(1, m_max - c_max - d_max)
            c_score = round(m_awarded * 0.4, 1)
            d_score = round(m_awarded * 0.3, 1)
            e_score = round(m_awarded - c_score - d_score, 1)
            rb = {
                "concept_score": c_score, "concept_max": c_max,
                "diagram_score": d_score, "diagram_max": d_max,
                "explanation_score": e_score, "explanation_max": e_max
            }
        else:
            try:
                c_max = int(float(rb.get("concept_max", round(m_max * 0.4))))
                d_max = int(float(rb.get("diagram_max", round(m_max * 0.3))))
                e_max = int(float(rb.get("explanation_max", m_max - c_max - d_max)))
            except (ValueError, TypeError):
                c_max = max(1, round(m_max * 0.4))
                d_max = max(1, round(m_max * 0.3))
                e_max = max(1, m_max - c_max - d_max)

            if c_max + d_max + e_max != m_max:
                e_max = max(0, m_max - c_max - d_max)

            try:
                c_score = float(rb.get("concept_score", 0.0))
                d_score = float(rb.get("diagram_score", 0.0))
                e_score = float(rb.get("explanation_score", 0.0))
            except (ValueError, TypeError):
                c_score = round(m_awarded * 0.4, 1)
                d_score = round(m_awarded * 0.3, 1)
                e_score = round(m_awarded - c_score - d_score, 1)

            rubric_sum = round(c_score + d_score + e_score, 1)
            if rubric_sum != m_awarded:
                rem = round(m_awarded - (c_score + d_score), 1)
                if 0 <= rem <= e_max:
                    e_score = rem
                else:
                    c_score = round(m_awarded * 0.4, 1)
                    d_score = round(m_awarded * 0.3, 1)
                    e_score = round(m_awarded - c_score - d_score, 1)

            rb["concept_score"] = round(c_score, 1)
            rb["concept_max"] = c_max
            rb["diagram_score"] = round(d_score, 1)
            rb["diagram_max"] = d_max
            rb["explanation_score"] = round(e_score, 1)
            rb["explanation_max"] = e_max

        q["rubric_breakdown"] = rb
        if "is_considered" not in q:
            q["is_considered"] = True

        total_awarded += m_awarded
        total_max += m_max

    total_awarded = round(total_awarded, 1)
    data["total_awarded_marks"] = total_awarded
    data["total_max_marks"] = total_max if total_max > 0 else int(data.get("total_max_marks", 10))
    data["attempted_total_marks"] = total_awarded
    data["attempted_max_marks"] = data["total_max_marks"]

    # Compute deterministic percentage and VTU Grade category
    pct = (total_awarded / data["total_max_marks"] * 100) if data["total_max_marks"] > 0 else 0
    if pct >= 70:
        data["vtu_grade"] = "First Class with Distinction (FCD)"
    elif pct >= 60:
        data["vtu_grade"] = "First Class (FC)"
    elif pct >= 50:
        data["vtu_grade"] = "Second Class (SC)"
    elif pct >= 40:
        data["vtu_grade"] = "Pass Class"
    else:
        data["vtu_grade"] = "Fail"

    return data

def evaluate_answer(question, student_answer, reference_context, subject):
    """
    Evaluates the student's handwritten answer against the ground truth using Gemini API.
    Returns a dictionary parsed from the API's JSON output.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    # Mapping for prompt context
    subject_names = {
        "dbms": "Database Management Systems (DBMS)",
        "cn": "Computer Networks (CN)",
        "dsa": "Data Structures and Algorithms (DSA)",
        "bcs703": "Cryptography & Network Security (BCS703 / BCA703)",
        "bca703": "Cryptography & Network Security (BCS703 / BCA703)"
    }
    
    full_subject_name = subject_names.get(subject.lower(), subject)
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are a fair, generous, and empathetic university examiner evaluating {full_subject_name} engineering answers.
You are given the extracted text from a question paper, the ground truth reference text from the curriculum, and the student's handwritten answer.

Question Paper (OCR Extracted):
{question}

Ground Truth Reference (Curriculum Benchmark):
{reference_context}

Student Answer (OCR Extracted):
{student_answer}

Task:
Evaluate the student's entire answer submission against the provided Question Paper. 

CRITICAL RULE 0 - SUBJECT VALIDATION:
Before evaluating, verify if the Question Paper and Answer Script match the selected subject: **{full_subject_name}**. If the documents are clearly about a completely different subject (e.g., they uploaded Computer Networks, but selected DBMS), you MUST reject it immediately! Return 0 for all marks, an empty question_breakdown array, and state in the general_feedback: "SUBJECT MISMATCH ERROR: You selected {full_subject_name}, but uploaded documents for a different subject. Evaluation aborted."

CRITICAL RULE 1 — SEMANTIC EQUIVALENCE & INTENT OVER VERBATIM KEYWORDS:
- The Ground Truth Reference is an instructor's topical benchmark, NOT a rigid keyword matching key.
- Real students write under exam stress and explain engineering mechanisms in their own words, everyday language, or simplified analogies.
- NEVER deduct marks or penalize a student simply because specific textbook phrases or formal keywords are missing, as long as the explanation demonstrates that the student understood the core concept, mechanism, or algorithm.
- Treat conceptually sound paraphrasing and student terminology with the same value as formal textbook terms.

Use the following Human-Like Generous Scoring Model:
1. Question Mapping: Identify which questions from the Question Paper the student is answering.
2. Generous Partial Marking: Be highly generous with partial marks. If the student demonstrates a general conceptual understanding or core intuition, award 70-90% of the marks for that question. Do NOT give a 0 unless the answer is completely blank or totally irrelevant.
3. OCR & Handwriting Forgiveness: Handwritten OCR often misses words, jumbles letters, or fails to read diagrams. Actively infer the student's intent. If a sentence is fragmented, assume the student wrote the correct concept, and give them the benefit of the doubt.
4. Conceptual Reward: Reward the student for trying. Explaining a concept in their own voice is a sign of true understanding; award full credit for that concept.
5. Diagram Intent: If the question asks for a diagram and there are scattered labels, boxes, or arrows, assume the student drew a valid schematic and award full or high diagram marks (80-100%).
6. Question-by-Question Breakdown: Evaluate each question individually. Provide awarded_marks, max_marks, and a positive, encouraging reason explaining why marks were awarded.
7. Irrelevant/Unreadable Rejection: Only return 0 marks if the text is literally complete gibberish or a completely different subject.

Produce an evaluation report with exactly these JSON keys:
1. "total_awarded_marks": A realistic total score (can be decimal).
2. "total_max_marks": The integer value of the total maximum marks for the questions attempted.
3. "question_breakdown": An array of objects. Each object must have "question_text" (string), "awarded_marks" (number), "max_marks" (integer), and "reasoning" (detailed string explaining why marks were deducted or given).
4. "strengths": An array of strings detailing what the student got right overall.
5. "missing_concepts": An array of strings detailing what core concepts or steps were missing overall.
6. "general_feedback": A concluding paragraph of overall feedback.
"""

    # Uncrowded, ultra-reliable models that do not hit 503 limits:
    models_to_try = ['gemini-3-flash-preview', 'gemini-3.1-flash-lite', 'gemini-3.5-flash']
    last_error = None
    
    for model_name in models_to_try:
        try:
            print(f"Trying model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "total_awarded_marks": {"type": "NUMBER"},
                            "total_max_marks": {"type": "INTEGER"},
                            "question_breakdown": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "question_text": {"type": "STRING"},
                                        "awarded_marks": {"type": "NUMBER"},
                                        "max_marks": {"type": "INTEGER"},
                                        "reasoning": {"type": "STRING"}
                                    },
                                    "required": ["question_text", "awarded_marks", "max_marks", "reasoning"]
                                }
                            },
                            "strengths": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"}
                            },
                            "missing_concepts": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"}
                            },
                            "general_feedback": {"type": "STRING"}
                        },
                        "required": ["total_awarded_marks", "total_max_marks", "question_breakdown", "strengths", "missing_concepts", "general_feedback"]
                    }
                )
            )
            
            # Parse and sanitize the JSON response
            print(f"Success with model: {model_name}")
            return sanitize_evaluation_data(json.loads(response.text))
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            print(f"Model {model_name} failed: {error_str[:100]}")
            # If it's a 503 (overloaded) or 429, wait briefly then try next model
            if '503' in error_str or 'UNAVAILABLE' in error_str or '429' in error_str:
                import time
                time.sleep(1.5)
                continue
            # If not found, try next model immediately
            if '404' in error_str or 'not found' in error_str.lower():
                continue
            break
    
    return {
        "total_awarded_marks": 0,
        "total_max_marks": 10,
        "question_breakdown": [],
        "strengths": [],
        "missing_concepts": [],
        "general_feedback": f"Evaluation failed: {str(last_error)}"
    }

def _render_file_to_jpeg_parts(file_path, is_answer_booklet=False, max_pages=3):
    """
    Renders an image or PDF into a list of Gemini types.Part JPEG objects.
    Skips front cover pages if it's a large university exam booklet.
    """
    import cv2
    import numpy as np
    parts = []
    
    if file_path.lower().endswith('.pdf'):
        import fitz
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        if is_answer_booklet and total_pages > 4:
            start_p = 5  # Skip cover & instruction sheets
            end_p = min(total_pages, start_p + max_pages)
        else:
            start_p = 0
            end_p = min(total_pages, max_pages)
            
        for p in range(start_p, end_p):
            pix = doc[p].get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif pix.n == 1: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
        doc.close()
    else:
        img = cv2.imread(file_path)
        if img is not None:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
            
    return parts

def evaluate_direct_multimodal(q_filepath, a_filepath, subject, reference_context="", strictness="generous"):
    """
    Direct Full-Booklet Multimodal Evaluation:
    Uploads documents to Google Cloud Files API and evaluates the ENTIRE booklet
    from Page 1 to the end, grading ALL attempted questions and diagrams simultaneously.
    Includes page citations, evidence quotes, rubric breakdowns, strictness settings, and VTU grading.
    Calibrated for SEMANTIC EQUIVALENCE & EMPATHETIC HUMAN EVALUATION (never penalizes for student phrasing).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
        
    subject_names = {
        "dbms": "Database Management Systems (DBMS)",
        "cn": "Computer Networks (CN)",
        "dsa": "Data Structures and Algorithms (DSA)",
        "bcs703": "Cryptography & Network Security (BCS703 / BCA703)",
        "bca703": "Cryptography & Network Security (BCS703 / BCA703)"
    }
    full_subject_name = subject_names.get(subject.lower(), subject)
    client = genai.Client(api_key=api_key)
    
    # Strictness directives - calibrated for semantic understanding and empathetic human scoring
    strictness_guidelines = {
        "generous": (
            "EVALUATION RIGOR: EMPATHETIC / HIGH GENEROSITY (Recommended for Handwritten Student Scripts).\n"
            "- Core Principle: Prioritize student intent, intuitive logic, and conceptual understanding above all else.\n"
            "- Semantic Equivalence: Fully accept informal wording, simplified everyday analogies, and student's own vocabulary. Award 80-100% partial marks whenever the core intuition or mechanism is captured.\n"
            "- Zero Robotic Keyword Penalties: NEVER deduct marks simply because specific textbook phrases or formal keywords are missing if the explanation demonstrates that the student understood the concept.\n"
            "- Diagrams: Award 85-100% diagram marks for any recognizable schematic, block diagram, or flowchart showing key components and flow, even if hand-drawn roughly without a ruler.\n"
            "- Vision & OCR Grace: Give maximum benefit of the doubt for handwriting irregularities, faint ink, or minor spelling slips."
        ),
        "standard": (
            "EVALUATION RIGOR: STANDARD IA (Balanced University Assessment - Concept First).\n"
            "- Core Principle: Value semantic equivalence and clear engineering understanding. Award 65-90% partial marks for solid conceptual attempts.\n"
            "- Semantic Understanding over Keyword Matching: Ground Truth is a conceptual benchmark, NOT a verbatim string-matching key. If the student conveys the working principle in their own terms, award high concept marks.\n"
            "- Diagrams: Award 75-90% diagram marks if core entities and directional relationships are shown.\n"
            "- Only deduct marks for genuinely incorrect technical statements, fatal logical contradictions, or completely omitted requirements."
        ),
        "strict": (
            "EVALUATION RIGOR: EXTERNAL BOARD STANDARD (Technical Rigor with Fair Semantic Recognition).\n"
            "- Core Principle: Maintain formal academic examination standards while still respecting student expression.\n"
            "- Accept accurate paraphrasing and conceptual descriptions (award 60-80% for sound partial explanations).\n"
            "- Diagrams should have clear entity blocks and logical connections (award 70-85% for correct structural attempts).\n"
            "- Deduct marks only where core technical mechanisms are substantively missing or incorrect, NOT for stylistic phrasing differences."
        )
    }
    rigor_directive = strictness_guidelines.get(strictness.lower(), strictness_guidelines["generous"])

    uploaded_files = []
    try:
        print("Uploading question paper and full answer booklet to cloud...")
        q_file = client.files.upload(file=q_filepath)
        uploaded_files.append(q_file)
        
        a_file = client.files.upload(file=a_filepath)
        uploaded_files.append(a_file)
        
        prompt = f"""
You are a senior university chief examiner evaluating real handwritten answer booklets for {full_subject_name} engineering examinations.
You are provided with:
1. The official Question Paper document.
2. The student's COMPLETE handwritten answer booklet (all pages).

Ground Truth Reference (Curriculum Benchmark):
{reference_context}

{rigor_directive}

CRITICAL EXAMINER DIRECTIVE — SEMANTIC EQUIVALENCE & EMPATHETIC HUMAN EVALUATION:
1. SEMANTIC EQUIVALENCE OVER EXACT KEYWORDS:
   - The Ground Truth Reference above is a topical benchmark, NOT a verbatim string-matching answer key.
   - Real students write under exam stress and time pressure; they naturally explain technical concepts in their own words, everyday phrasing, or simplified descriptions.
   - NEVER penalize a student or deduct marks simply because they omitted specific textbook jargon or verbatim phrasing, as long as their explanation communicates the correct conceptual mechanism, algorithm, or intuition.
   - Treat conceptually accurate paraphrases, synonyms, and engineering intuition with the same validity as formal textbook sentences.
2. GENEROUS PARTIAL CREDIT FOR CONCEPTUAL GRASP:
   - Look for reasons to AWARD marks based on what the student understood, rather than excuses to deduct marks.
   - If a student demonstrates the general intuition or core mechanism of a question, award 70%–90% partial marks.
   - Only award 0 marks if a question is completely left blank or the response is entirely irrelevant/gibberish.
3. DIAGRAM INTENT & SCHEMATIC EFFORT:
   - Engineering students hand-draw diagrams under exam constraints.
   - If a diagram includes the primary blocks, arrows, or structural flow—even if hand-drawn without a ruler, slightly uneven, or with informal labels—award 75%–100% of the diagram marks.
4. HANDWRITING & VISUAL TOLERANCE:
   - Answer scripts are handwritten. Grant the student the benefit of the doubt on faint ink, slanted handwriting, or minor spelling slips.

TASK:
1. Scan through the ENTIRE answer booklet from Page 1 to the very last page.
2. Identify EVERY SINGLE QUESTION attempted by the student across all pages (e.g., Q1.a, Q8.a, Q8.b, Q6.b, Q6.a, Q2.b, Q3.b, etc.).
3. For EACH question attempted:
   - Identify the exact question text and number from the Question Paper.
   - Record the exact Page Number(s) where the student wrote this answer (e.g. "Page 6", "Pages 8 - 9").
   - Extract a direct Evidence Quote or concise transcription of handwriting / diagram description found on that page (e.g., 'Drawn 3-way TCP handshake with SYN, SYN+ACK, ACK flags').
   - Provide a Detailed 3-Part Rubric Breakdown:
     * Concept / Definition score (concept_score out of concept_max)
     * Diagram / Schematic score (diagram_score out of diagram_max)
     * Working Principle / Explanation score (explanation_score out of explanation_max)
     * Note: concept_score + diagram_score + explanation_score MUST equal awarded_marks.
     * concept_max + diagram_max + explanation_max MUST equal max_marks.
   - Provide clear, encouraging reasoning highlighting what the student explained well, giving positive feedback and friendly suggestions for further study.
4. Calculate the total awarded marks and total maximum marks for all attempted questions.
5. Determine the student's VTU Grade category:
   - "First Class with Distinction (FCD)" if percentage >= 70%
   - "First Class (FC)" if 60% <= percentage < 70%
   - "Second Class (SC)" if 50% <= percentage < 60%
   - "Pass Class" if 40% <= percentage < 50%
   - "Fail" if percentage < 40%
6. Provide overall strengths, missing concepts (as constructive study tips), and encouraging general feedback.
7. Provide a concise transcription summary of all attempted answers in "transcribed_summary".

SUBJECT INTELLIGENCE: Always evaluate according to the actual subject shown on the Question Paper document. If the detected subject differs from the selected option ({full_subject_name}), adapt and evaluate the paper based on the Question Paper, noting the detected subject in "detected_subject". Never abort with 0 marks due to a dropdown mismatch.
"""

        schema = {
            "type": "OBJECT",
            "properties": {
                "detected_subject": {"type": "STRING"},
                "vtu_grade": {"type": "STRING"},
                "transcribed_summary": {"type": "STRING"},
                "total_awarded_marks": {"type": "NUMBER"},
                "total_max_marks": {"type": "INTEGER"},
                "question_breakdown": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "question_text": {"type": "STRING"},
                            "page_citations": {"type": "STRING"},
                            "evidence_quote": {"type": "STRING"},
                            "awarded_marks": {"type": "NUMBER"},
                            "max_marks": {"type": "INTEGER"},
                            "rubric_breakdown": {
                                "type": "OBJECT",
                                "properties": {
                                    "concept_score": {"type": "NUMBER"},
                                    "concept_max": {"type": "INTEGER"},
                                    "diagram_score": {"type": "NUMBER"},
                                    "diagram_max": {"type": "INTEGER"},
                                    "explanation_score": {"type": "NUMBER"},
                                    "explanation_max": {"type": "INTEGER"}
                                },
                                "required": ["concept_score", "concept_max", "diagram_score", "diagram_max", "explanation_score", "explanation_max"]
                            },
                            "reasoning": {"type": "STRING"}
                        },
                        "required": ["question_text", "page_citations", "evidence_quote", "awarded_marks", "max_marks", "rubric_breakdown", "reasoning"]
                    }
                },
                "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
                "missing_concepts": {"type": "ARRAY", "items": {"type": "STRING"}},
                "general_feedback": {"type": "STRING"}
            },
            "required": ["detected_subject", "vtu_grade", "transcribed_summary", "total_awarded_marks", "total_max_marks", "question_breakdown", "strengths", "missing_concepts", "general_feedback"]
        }

        models = ['gemini-3-flash-preview', 'gemini-3.1-flash-lite', 'gemini-3.5-flash']
        last_err = None
        
        for m in models:
            try:
                print(f"Full-Booklet evaluation with model {m} (Mode: {strictness})...")
                resp = client.models.generate_content(
                    model=m,
                    contents=[q_file, a_file, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
                )
                data = json.loads(resp.text)
                data = sanitize_evaluation_data(data)
                print(f"Full-Booklet evaluation SUCCESS with {m}! Found {len(data.get('question_breakdown', []))} questions. Total Marks: {data['total_awarded_marks']}/{data['total_max_marks']}")
                return data
            except Exception as e:
                last_err = e
                print(f"Model {m} failed: {e}")
                continue
                
        return {
            "detected_subject": full_subject_name,
            "vtu_grade": "Pending Evaluation",
            "transcribed_summary": "Full booklet evaluation incomplete.",
            "total_awarded_marks": 0,
            "total_max_marks": 10,
            "question_breakdown": [],
            "strengths": [],
            "missing_concepts": [],
            "general_feedback": f"Evaluation service temporarily busy: {str(last_err)}"
        }
    finally:
        for uf in uploaded_files:
            try:
                client.files.delete(name=uf.name)
            except Exception:
                pass

def explain_grade_query(question_text, evidence_quote, current_reasoning, student_query, subject="Engineering"):
    """
    Interactive 'Ask the Examiner' clarification assistant:
    Provides an encouraging, syllabus-accurate explanation directly answering the student's question.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a warm, encouraging university professor and mentor for {subject}.
A student has reviewed their evaluated answer and has a question about their marks.

Question: {question_text}
Student's Written Evidence / Work: {evidence_quote}
Examiner's Grading Reasoning: {current_reasoning}

Student's Clarification Query:
"{student_query}"

Provide a concise (2-4 sentences), encouraging, and empathetic response:
1. Reassure the student, affirming the conceptual understanding and intuition they demonstrated in their answer.
2. Provide a constructive, actionable tip on what minor addition (e.g. an extra diagram label, key formula step, or technical nuance) will help them secure full 100% marks in their next exam.
Keep the tone warm, respectful, and motivating.
"""
    try:
        resp = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt
        )
        return resp.text.strip()
    except Exception as e:
        return f"Examiner feedback unavailable: {str(e)}"

def apply_vtu_module_rules(eval_data):
    """
    VTU Choice & Module Tagging Engine:
    - Extracts precise question numbers without false positives from technical terms (e.g. 3-way, 2PL, IPv4).
    - Preserves 100% of attempted questions by default: all questions count towards the total score.
    - Only flags true internal choices if a 10-question external scheme is explicitly detected (Q1..Q10 with Q > 5).
    - NEVER corrupts or silently deletes attempted student marks.
    """
    import re
    
    questions = eval_data.get('question_breakdown', [])
    if not questions:
        return eval_data
        
    def extract_main_q(text):
        text = str(text or '').strip()
        # 1. Match explicit "Q1", "Question 1", "Q. 1", "Q 1" at the start
        m = re.match(r'^(?:Q(?:uestion)?\.?\s*|\bQ\s*)(\d{1,2})\b', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # 2. Match leading numbering e.g. "1.", "1.a", "1(a)", "1 -", "1)"
        m = re.match(r'^(\d{1,2})\s*[\.\)\-a-zA-Z\s]', text)
        if m:
            return int(m.group(1))
        # 3. Match Q prefix within first 25 characters (e.g. "[Q1.a]", "(Question 2)")
        m = re.search(r'\b(?:Q|Question)\.?\s*(\d{1,2})\b', text[:25], re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 0

    mq_list = [extract_main_q(q.get('question_text', '')) for q in questions]
    valid_mqs = [mq for mq in mq_list if mq > 0]
    max_mq = max(valid_mqs) if valid_mqs else 0
    # A true 10-question VTU external paper has questions up to Q6..Q10 (5 modules x 2 questions)
    is_external_10q_choice_paper = (max_mq > 5)

    main_q_map = {}
    for idx, q in enumerate(questions):
        mq = mq_list[idx]
        q['main_q_num'] = mq
        
        if is_external_10q_choice_paper:
            mod_num = (mq + 1) // 2 if (1 <= mq <= 10) else ((idx % 5) + 1)
        else:
            mod_num = mq if (1 <= mq <= 5) else (idx + 1)

        q['module_num'] = mod_num
        q['is_considered'] = True
        q['module_status'] = f"Module {mod_num} Attempted" if mod_num > 0 else "Attempted"

        if mod_num not in main_q_map:
            main_q_map[mod_num] = {}
        if mq not in main_q_map[mod_num]:
            main_q_map[mod_num][mq] = []
        main_q_map[mod_num][mq].append(q)

    choice_applied = False
    if is_external_10q_choice_paper:
        for mod_num, mqs in main_q_map.items():
            if mod_num == 0 or len(mqs) <= 1:
                continue
            choice_applied = True
            scores = {}
            for mq, q_list in mqs.items():
                scores[mq] = sum(q.get('awarded_marks', 0) for q in q_list)
            best_mq = max(scores, key=scores.get)
            for mq, q_list in mqs.items():
                if mq != best_mq:
                    for q in q_list:
                        q['is_considered'] = False
                        q['module_status'] = f"Extra Attempt in Module {mod_num} (Score: {scores[mq]} vs Q{best_mq}: {scores[best_mq]} - Excluded from Official Total)"
                else:
                    for q in q_list:
                        q['is_considered'] = True
                        q['module_status'] = f"Module {mod_num} Choice Counted (Best of Module)"

    attempted_total = round(sum(float(q.get('awarded_marks', 0)) for q in questions), 1)
    attempted_max = sum(int(q.get('max_marks', 0)) for q in questions)
    official_total = round(sum(float(q.get('awarded_marks', 0)) for q in questions if q.get('is_considered', True)), 1)
    official_max = sum(int(q.get('max_marks', 0)) for q in questions if q.get('is_considered', True))

    eval_data['attempted_total_marks'] = attempted_total
    eval_data['attempted_max_marks'] = attempted_max
    eval_data['vtu_official_marks'] = official_total
    eval_data['vtu_official_max'] = official_max
    eval_data['vtu_choice_applied'] = choice_applied

    # Total awarded marks and max marks:
    eval_data['total_awarded_marks'] = official_total if choice_applied else attempted_total
    eval_data['total_max_marks'] = official_max if choice_applied else attempted_max

    # Recalculate percentage and VTU Grade
    effective_total = eval_data['total_awarded_marks']
    effective_max = eval_data['total_max_marks']
    percent = (effective_total / effective_max * 100) if effective_max > 0 else 0
    if percent >= 70:
        eval_data['vtu_grade'] = "First Class with Distinction (FCD)"
    elif percent >= 60:
        eval_data['vtu_grade'] = "First Class (FC)"
    elif percent >= 50:
        eval_data['vtu_grade'] = "Second Class (SC)"
    elif percent >= 40:
        eval_data['vtu_grade'] = "Pass Class"
    else:
        eval_data['vtu_grade'] = "Fail"

    return eval_data
