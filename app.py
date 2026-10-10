import os
import uuid
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from ocr_engine import extract_text
from knowledge_engine import retrieve_context, retrieve_dynamic_context
from evaluator import evaluate_answer, evaluate_direct_multimodal, explain_grade_query, apply_vtu_module_rules

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "model": "gemini-3-flash-preview"})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    start_time = time.time()
    if 'question_image' not in request.files or 'answer_image' not in request.files:
        return jsonify({"error": "Both Question Paper and Answer Sheet files are required."}), 400
        
    q_file = request.files['question_image']
    a_file = request.files['answer_image']
    
    if q_file.filename == '' or a_file.filename == '':
        return jsonify({"error": "Please select files for both Question and Answer."}), 400
        
    if not allowed_file(q_file.filename) or not allowed_file(a_file.filename):
        return jsonify({"error": "Invalid file type. Only PNG, JPG, JPEG, PDF allowed."}), 400
        
    subject = request.form.get('subject', 'dbms')
    strictness = request.form.get('strictness', 'standard')
    
    try:
        # 1. Save uploaded images safely
        q_ext = q_file.filename.rsplit('.', 1)[1].lower()
        a_ext = a_file.filename.rsplit('.', 1)[1].lower()
        
        q_filename = f"q_{uuid.uuid4().hex}.{q_ext}"
        a_filename = f"a_{uuid.uuid4().hex}.{a_ext}"
        
        q_filepath = os.path.join(app.config['UPLOAD_FOLDER'], q_filename)
        a_filepath = os.path.join(app.config['UPLOAD_FOLDER'], a_filename)
        
        q_file.save(q_filepath)
        a_file.save(a_filepath)
        
        # 2. Query-Aware Dynamic RAG from ChromaDB
        reference_context = retrieve_dynamic_context(q_filepath, subject=subject)
        
        # 3. Direct 1-Step Multimodal Evaluation (~18-25s)
        # Evaluates handwriting, diagrams, questions & allocates marks in 1 step
        evaluation_report = evaluate_direct_multimodal(
            q_filepath=q_filepath,
            a_filepath=a_filepath,
            subject=subject,
            reference_context=reference_context,
            strictness=strictness
        )
        
        # 4. Apply VTU 'Best-of-Module' Choice Rules Engine
        evaluation_report = apply_vtu_module_rules(evaluation_report)
        
        processing_time = round(time.time() - start_time, 2)
        
        # Build question and answer summaries for the Technical Details tab
        q_text_preview = "Extracted directly from Question Paper document."
        if evaluation_report.get("question_breakdown"):
            questions = [f"• {q.get('question_text')} [Max: {q.get('max_marks')} Marks]" for q in evaluation_report["question_breakdown"]]
            q_text_preview = "\n\n".join(questions)
            
        a_text_preview = evaluation_report.get("transcribed_summary", "Handwritten content evaluated visually by Multimodal AI.")
        
        response_data = {
            "evaluation": evaluation_report,
            "extracted_question": q_text_preview,
            "extracted_text": a_text_preview,
            "reference_context": reference_context,
            "q_image_filename": q_filename,
            "a_image_filename": a_filename,
            "processing_time_sec": processing_time
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "hint": "Ensure both uploaded documents are legible and Google API service is reachable."
        }), 500

@app.route('/ask_examiner', methods=['POST'])
def ask_examiner():
    data = request.get_json() or {}
    question_text = data.get('question_text', '')
    evidence_quote = data.get('evidence_quote', '')
    reasoning = data.get('reasoning', '')
    student_query = data.get('student_query', '')
    subject = data.get('subject', 'Engineering')
    
    if not student_query:
        return jsonify({"error": "Please provide a query for the examiner."}), 400
        
    answer = explain_grade_query(
        question_text=question_text,
        evidence_quote=evidence_quote,
        current_reasoning=reasoning,
        student_query=student_query,
        subject=subject
    )
    return jsonify({"answer": answer})

@app.route('/page_preview/<filename>/<int:page_num>')
def page_preview(filename, page_num):
    """
    Renders and serves high-res visual page for the interactive Split-Screen Inspector.
    """
    safe_fn = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_fn)
    if not os.path.exists(filepath):
        return jsonify({"error": "Document not found"}), 404
        
    if filepath.lower().endswith('.pdf'):
        try:
            import pymupdf
            doc = pymupdf.open(filepath)
            # Page num is 1-indexed from frontend
            p_idx = max(0, min(len(doc) - 1, page_num - 1))
            page = doc[p_idx]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("jpeg")
            doc.close()
            from flask import Response
            return Response(img_bytes, mimetype="image/jpeg")
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_fn)

# Route to serve uploaded images (if needed by frontend)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    import socket
    port = 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            port = 5001

    print("\n" + "=" * 62)
    print("  🚀 AI Exam Evaluation & Grading System")
    print(f"  👉 Server active! Open in browser: http://127.0.0.1:{port}")
    print("=" * 62 + "\n")
    app.run(host='127.0.0.1', port=port, debug=True)
