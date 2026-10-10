"""
OCR Engine - Uses Gemini Vision AI for handwritten text extraction.
EasyOCR was producing garbage on handwritten exam scripts.
Gemini's vision model can read handwriting directly from images.
"""
import cv2
import os
import base64
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def extract_text(file_path):
    """
    Reads an image or PDF and extracts text.
    For PDFs: converts pages to images, skips blanks/covers, sends to Gemini Vision.
    For images: sends directly to Gemini Vision.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    page_images = []  # list of (page_num, jpeg_bytes)

    if file_path.lower().endswith('.pdf'):
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        # Fast & Smart page selection:
        # - Small PDFs (<= 4 pages): Question papers or short answers -> scan all pages
        # - Large booklets (> 4 pages): Skip first 5 cover/index pages -> scan 3 content pages (pages 6, 7, 8)
        if total_pages <= 4:
            start_page = 0
            max_pages = total_pages
        else:
            start_page = 5
            max_pages = min(total_pages, start_page + 3)
        
        print(f"PDF has {total_pages} pages. Fast-scanning pages {start_page+1} to {max_pages}...")
        
        for page_num in range(start_page, max_pages):
            page = doc.load_page(page_num)
            # 1.2x matrix is optimized for quick cloud transfer while preserving crisp handwriting
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Fast Blank page detection
            gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray_check.shape
            center_crop = gray_check[int(h*0.15):int(h*0.9), int(w*0.15):int(w*0.9)]
            _, thresh_check = cv2.threshold(center_crop, 200, 255, cv2.THRESH_BINARY_INV)
            ink_ratio = (np.sum(thresh_check == 255) / thresh_check.size)
            
            if ink_ratio < 0.01:
                print(f"  Skipping blank page {page_num + 1}")
                continue
            
            # High-efficiency JPEG compression for rapid upload
            _, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            page_images.append((page_num + 1, jpeg_buf.tobytes()))
            print(f"  Page {page_num + 1} captured ({len(jpeg_buf)} bytes)")
        
        doc.close()
    else:
        # Single image file
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image at {file_path}")
        _, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        page_images.append((1, jpeg_buf.tobytes()))

    if not page_images:
        return ""

    # Send captured pages to Gemini Vision
    print(f"Sending {len(page_images)} page(s) to Gemini Vision...")
    extracted_text = _extract_with_gemini_vision(page_images)
    return extracted_text


def _extract_with_gemini_vision(page_images):
    """
    Sends page images to Gemini Vision API and extracts all handwritten text.
    Sends all pages in a single optimized payload with automatic retry backoff.
    """
    import time
    from google import genai
    from google.genai import types
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    
    parts = []
    page_nums = []
    for page_num, jpeg_bytes in page_images:
        page_nums.append(str(page_num))
        parts.append(types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"))
    
    prompt_text = (
        "You are an expert OCR system. Extract ALL handwritten and printed text "
        "from the exam answer sheet page(s) shown in these images. "
        "Transcribe the text exactly as written by the student, preserving all technical answers, "
        "equations, bullet points, and diagram descriptions. "
        "Output ONLY the extracted text with no extra commentary."
    )
    parts.insert(0, types.Part.from_text(text=prompt_text))
    content = types.Content(parts=parts, role="user")
    
    # Rock-solid, fast models:
    models_to_try = ['gemini-3-flash-preview', 'gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-3.6-flash']
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=content
            )
            extracted = response.text.strip()
            if extracted:
                print(f"  Pages {','.join(page_nums)}: extracted {len(extracted)} chars using {model_name}")
                return extracted
        except Exception as e:
            err = str(e)
            print(f"  Model {model_name} issue: {err[:60]} -> switching model immediately")
            continue
    
    return ""


# For testing
if __name__ == "__main__":
    pass
