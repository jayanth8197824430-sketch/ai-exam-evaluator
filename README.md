# 🎓 Autonomous Multimodal AI Exam Evaluation & Grading System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-Multimodal_Vision-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Database-FF6F00?style=for-the-badge)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An intelligent, transparent, and university-grade examination assessment platform engineered for handwritten university answer booklets (specifically calibrated for **VTU CBCS 5-Module Scheme**). 

Unlike legacy systems that rely on error-prone OCR pipelines or generic \'black-box\' LLM grading, this system uses **direct multimodal visual reasoning**, **query-aware Dynamic RAG**, and **human-in-the-loop audit tools** to make evaluation accurate, defensible, and instantaneous.

---

## 🌟 Key Features

### 🧠 1. Native Multimodal Visual Evaluation
- Directly processes multi-page handwritten booklets and question papers via **Google Gemini Multimodal Vision API**.
- Accurately grades cursive handwriting, technical block diagrams, schemas, truth tables, and mathematical derivations without lossy OCR preprocessing.

### ⚡ 2. Neuro-Symbolic Hybrid RAG (Dense Embeddings + BM25 RRF)
- Combines **Dense Vector Semantic Search** (`all-MiniLM-L6-v2` in ChromaDB) with **Sparse Lexical Search (Okapi BM25)**.
- Uses **Reciprocal Rank Fusion (RRF)** to balance conceptual understanding with exact technical acronyms (e.g. `AES`, `IPSec`, `ESP`, `B+ Tree`, `2PL`), guaranteeing syllabus-accurate reference context.

### 🔬 3. Transparent Evidence Citations & Split-Screen Inspector
- **Page Citations**: Pinpoints the exact booklet page (e.g., `📍 Pages 8 – 9`) where each answer was found.
- **Evidence Quotes**: Excerpts the student's exact handwritten phrasing.
- **Split-Screen Inspector**: Allows examiners to inspect the high-resolution rendered page image side-by-side with the rubric breakdown and AI reasoning.

### ⚖️ 4. Symbolic VTU 'Best-of-Module' Choice Engine
- Normalizes university CBCS choice rules (e.g., *Answer Q1 OR Q2 in Module 1*).
- If a candidate attempts both, it awards marks to the higher-scoring attempt, tags the other as *Extra Attempt (Excluded)*, and computes both **VTU Official Marks** and **Attempted Total**.

### 👨‍🏫 5. Human-in-the-Loop Teacher Override
- Ethical AI design: instructors retain ultimate authority to adjust marks and append remarks inline.
- Instant recalculation of total marks, percentages, and VTU Grade Classes (*Distinction, First Class, Pass, Fail*).

### 📄 6. 1-Click Official Marksheet Export
- Produces a printable university grade card with institutional headers, candidate details, module-wise breakdown, and digital verification watermarks.

### 💬 7. Interactive Student 'Explain My Grade' Assistant
- Embedded mini-chat powered by `gemini-3.1-flash-lite` allowing students to query mark deductions and receive syllabus-accurate advice.

### 💖 8. Semantic Equivalence & Empathetic Human Grading Engine
- **Concept Over Verbatim Keywords**: The syllabus ground truth is treated as a conceptual guide, not a robotic string-matching checklist. Real students expressing engineering mechanisms in their own words or simplified analogies receive full/high partial credit (75%–100%).
- **Diagram Effort Recognition**: Hand-drawn schematics and block diagrams showing correct structural flow receive generous diagram marks, forgiving hand-drawn imperfections.

### 🔥 9. Computer Vision Scribe & Legibility Heatmap
- Embedded OpenCV Sobel spatial gradient filter calculating a continuous **Legibility Index (%)** to flag dense cursive, faint handwriting, or scribbles for human review.

---

## 🛠️ Architecture Overview

```text
[ Question Paper PDF ] + [ Handwritten Answer Booklet PDF ]
               │                          │
               ├──────────────────────────┤
               ▼                          ▼
     [ Question Prompts ]       [ Visual Page Inspector ]
               │                          │
               ▼                          │
  ┌─────────────────────────────────────┐ │
  │    Neuro-Symbolic Hybrid RAG        │ │
  │  • Dense Semantic (ChromaDB)        │ │
  │  • Sparse Lexical (Okapi BM25)      │ │
  │  • Reciprocal Rank Fusion (RRF)     │ │
  │  (3,300+ Syllabus Chunks Indexed)   │ │
  └──────────────────┬──────────────────┘ │
                     │                    │
                     ▼                    ▼
  ┌─────────────────────────────────────────────────────┐
  │   Google Gemini Multimodal Reasoning Engine         │
  │   - Handwriting Analysis & Layout Parsing           │
  │   - Diagram & Proof Verification                    │
  │   - Page Citation & Evidence Extraction             │
  │   - 3-Part Rubric Scoring (Concept / Diagram / Work)│
  └──────────────────────────┬──────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────┐
  │         Symbolic VTU CBCS Rule Engine               │
  │   - 5-Module Choice Normalization (Best of Q1/Q2)   │
  │   - Deterministic Arithmetic & Grade Classification │
  └──────────────────────────┬──────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────┐
  │            Cyber-Glassmorphism Web Dashboard        │
  │   • Split-Screen Inspector Mode                     │
  │   • Inline Teacher Override & Recalculation         │
  │   • 1-Click VTU Marksheet PDF Generator             │
  │   • 'Ask Examiner' Clarification Assistant         │
  └─────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
`ash
git clone https://github.com/YOUR_USERNAME/ai-exam-evaluator.git
cd ai-exam-evaluator
`

### 2. Set Up Virtual Environment & Dependencies
`ash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
`

### 3. Configure Gemini API Key
Create a .env file in the root directory:
`env
GEMINI_API_KEY=your_api_key_here
`
*(Get a free API key at [Google AI Studio](https://aistudio.google.com/))*

### 4. Run the Application
`ash
python app.py
`
Open **http://127.0.0.1:5000** in your browser.

---

## 📂 Project Structure

```text
├── app.py                  # Main Flask application & API routes
├── evaluator.py            # Gemini Multimodal engine & VTU Choice Rule Engine
├── knowledge_engine.py     # ChromaDB Dynamic RAG retriever (offline-cached)
├── ocr_engine.py           # PyMuPDF visual page renderer & image preprocessor
├── chroma_db/              # Persistent ChromaDB vector database (3,000+ chunks)
├── templates/
│   └── index.html          # Cyberpunk glassmorphism UI & Inspector Modal
├── uploads/                # Temporary document storage
├── .env.example            # Environment template
└── requirements.txt        # Python dependencies
```

---

## 👨‍💻 Author & Developer
- **Jayanth**
- GitHub: [@jayanth8197824430-sketch](https://github.com/jayanth8197824430-sketch)
- Email: `jayanth8197824430@gmail.com`

---

## 📜 License
This project is licensed under the MIT License.
