import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

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
        "dsa": "Data Structures and Algorithms (DSA)"
    }
    
    full_subject_name = subject_names.get(subject.lower(), subject)
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are a fair, generous, and empathetic university examiner for {full_subject_name} exams.
You are given the extracted text from a question paper, the ground truth reference text from the textbook, and the student's OCR-extracted handwritten answer.

Question Paper (OCR Extracted):
{question}

Ground Truth Reference: {reference_context}

Student Answer (OCR Extracted):
{student_answer}

Task:
Evaluate the student's entire answer submission against the provided Question Paper. 

CRITICAL RULE 0 - SUBJECT VALIDATION:
Before evaluating, verify if the Question Paper and Answer Script match the selected subject: **{full_subject_name}**. If the documents are clearly about a completely different subject (e.g., they uploaded Computer Networks, but selected DBMS), you MUST reject it immediately! Return 0 for all marks, an empty question_breakdown array, and state in the general_feedback: "SUBJECT MISMATCH ERROR: You selected {full_subject_name}, but uploaded documents for a different subject. Evaluation aborted."

Use the following Human-Like Generous Scoring Model:
1. Question Mapping: Identify which questions from the Question Paper the student is answering.
2. Generous Partial Marking (IMPORTANT): Be highly generous with partial marks. If the student demonstrates a general conceptual understanding, award at least 50-80% of the marks for that question, even if exact textbook keywords are missing. Do NOT give a 0 unless the answer is completely blank or totally irrelevant.
3. OCR Forgiveness: Handwritten OCR often misses words, jumbles letters, or fails to read diagrams. Actively infer the student's intent. If a sentence is fragmented, assume the student wrote the correct concept but the scanner missed it, and DO NOT penalize them.
4. Conceptual Reward: Reward the student for trying. If they explain a concept in their own words rather than textbook language, give them full credit for that concept.
5. Diagram Assumption: If the question requires a diagram and the OCR text contains scattered words that look like labels, assume the student drew a correct diagram and award full diagram marks.
6. Question-by-Question Breakdown: You MUST evaluate each question individually. Provide the marks awarded, maximum marks, and a positive, encouraging reason explaining why marks were awarded.
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
            
            # Parse the JSON response
            print(f"Success with model: {model_name}")
            return json.loads(response.text)
            
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

def evaluate_direct_multimodal(q_filepath, a_filepath, subject, reference_context="", strictness="standard"):
    """
    Direct Full-Booklet Multimodal Evaluation:
    Uploads documents to Google Cloud Files API and evaluates the ENTIRE booklet
    from Page 1 to the end, grading ALL attempted questions and diagrams simultaneously.
    Includes page citations, evidence quotes, rubric breakdowns, strictness settings, and VTU grading.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
        
    subject_names = {
        "dbms": "Database Management Systems (DBMS)",
        "cn": "Computer Networks (CN)",
        "dsa": "Data Structures and Algorithms (DSA)"
    }
    full_subject_name = subject_names.get(subject.lower(), subject)
    client = genai.Client(api_key=api_key)
    
    # Strictness directives
    strictness_guidelines = {
        "generous": (
            "EVALUATION RIGOR: GENEROUS (Formative Assessment / Practice Mode).\n"
            "- Be highly encouraging and empathetic. Award 70-90% partial marks for answers capturing core concepts.\n"
            "- Overlook minor handwriting imperfections, spelling errors, or slight formula omissions.\n"
            "- Award full diagram marks if a relevant schematic or labeled drawing is attempted."
        ),
        "standard": (
            "EVALUATION RIGOR: STANDARD (University Internal Assessment / IA Mode).\n"
            "- Balance conceptual understanding with technical accuracy. Award 50-80% partial marks.\n"
            "- Diagrams must have recognizable structure, correct labels, and clean directional flow.\n"
            "- Deduct partial marks for missing working steps or incomplete algorithm explanations."
        ),
        "strict": (
            "EVALUATION RIGOR: STRICT (VTU External Examination Board Standard).\n"
            "- Enforce rigorous university standards. Demand exact technical terminology, keywords, and accurate definitions.\n"
            "- Diagrams must be accurate with precise labels, interfaces, or protocol handshakes.\n"
            "- Award 0 marks for vague, non-technical generalizations or irrelevant statements."
        )
    }
    rigor_directive = strictness_guidelines.get(strictness.lower(), strictness_guidelines["standard"])

    uploaded_files = []
    try:
        print("Uploading question paper and full answer booklet to cloud...")
        q_file = client.files.upload(file=q_filepath)
        uploaded_files.append(q_file)
        
        a_file = client.files.upload(file=a_filepath)
        uploaded_files.append(a_file)
        
        prompt = f"""
You are a senior university chief examiner for {full_subject_name} engineering examinations.
You are provided with:
1. The official Question Paper document.
2. The student's COMPLETE handwritten answer booklet (all pages).

Ground Truth Reference from Curriculum:
{reference_context}

{rigor_directive}

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
   - Provide clear, encouraging reasoning explaining marks awarded and missing concepts.
4. Calculate the total awarded marks and total maximum marks for all attempted questions.
5. Determine the student's VTU Grade category:
   - "First Class with Distinction (FCD)" if percentage >= 70%
   - "First Class (FC)" if 60% <= percentage < 70%
   - "Second Class (SC)" if 50% <= percentage < 60%
   - "Pass Class" if 40% <= percentage < 50%
   - "Fail" if percentage < 40%
6. Provide overall strengths, missing concepts, and encouraging general feedback.
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
                print(f"Full-Booklet evaluation SUCCESS with {m}! Found {len(data.get('question_breakdown', []))} questions.")
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
You are a friendly, encouraging university professor and examiner for {subject}.
A student has reviewed their evaluated answer and has a question about their marks.

Question: {question_text}
Student's Written Evidence / Work: {evidence_quote}
Examiner's Grading Reasoning: {current_reasoning}

Student's Clarification Query:
"{student_query}"

Provide a concise (2-4 sentences), encouraging, and technically precise response explaining:
1. Exactly why the marks were awarded or where points were lost according to academic standards.
2. An actionable tip on how the student can score full marks next time (e.g., specific missing diagram labels, missing formula, or step).
Be polite, professional, and inspiring.
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
    VTU 'Best-of-Module' Choice Rule Engine:
    Under VTU schemes, question papers have 5 Modules (Q1/Q2, Q3/Q4, Q5/Q6, Q7/Q8, Q9/Q10).
    If a student answers both questions within the same module, university rules state
    that only the higher-scoring question counts towards the official marks.
    """
    import re
    
    questions = eval_data.get('question_breakdown', [])
    if not questions:
        return eval_data
        
    def extract_main_q(text):
        m = re.search(r'(?:Q(?:uestion)?\s*)?(\d{1,2})', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 0
        
    # Group sub-questions by main question
    main_q_map = {}
    for idx, q in enumerate(questions):
        mq = extract_main_q(q.get('question_text', ''))
        q['main_q_num'] = mq
        
        # Module assignment: 1..2 -> Mod 1, 3..4 -> Mod 2, 5..6 -> Mod 3, 7..8 -> Mod 4, 9..10 -> Mod 5
        if 1 <= mq <= 10:
            mod_num = (mq + 1) // 2
        else:
            mod_num = 0
        q['module_num'] = mod_num
        q['is_considered'] = True
        q['module_status'] = f"Module {mod_num} Attempted" if mod_num > 0 else "Attempted"
        
        if mod_num not in main_q_map:
            main_q_map[mod_num] = {}
        if mq not in main_q_map[mod_num]:
            main_q_map[mod_num][mq] = []
        main_q_map[mod_num][mq].append(q)
        
    # Check each module for internal choices (e.g. Q1 vs Q2, Q7 vs Q8)
    choice_applied = False
    for mod_num, mqs in main_q_map.items():
        if mod_num == 0 or len(mqs) <= 1:
            continue
            
        # Student attempted multiple main questions in the same module!
        choice_applied = True
        # Calculate sum of awarded marks for each main question in this module
        scores = {}
        for mq, q_list in mqs.items():
            scores[mq] = sum(q.get('awarded_marks', 0) for q in q_list)
            
        # Find best-scoring question
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
                    
    # Calculate official VTU score vs attempted score
    attempted_total = round(sum(q.get('awarded_marks', 0) for q in questions), 1)
    attempted_max = sum(q.get('max_marks', 0) for q in questions)
    
    official_total = round(sum(q.get('awarded_marks', 0) for q in questions if q.get('is_considered', True)), 1)
    official_max = sum(q.get('max_marks', 0) for q in questions if q.get('is_considered', True))
    
    eval_data['attempted_total_marks'] = attempted_total
    eval_data['attempted_max_marks'] = attempted_max
    eval_data['vtu_official_marks'] = official_total
    eval_data['vtu_official_max'] = official_max
    eval_data['vtu_choice_applied'] = choice_applied
    
    # If choice was applied, official awarded marks reflects the choice
    eval_data['total_awarded_marks'] = official_total
    eval_data['total_max_marks'] = official_max
    
    # Recalculate VTU Grade based on official marks
    percent = (official_total / official_max * 100) if official_max > 0 else 0
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
