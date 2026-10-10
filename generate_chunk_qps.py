import os
from PIL import Image, ImageDraw, ImageFont

output_dir = 'Question_Papers'
os.makedirs(output_dir, exist_ok=True)

# Load fonts
try:
    title_font = ImageFont.truetype("arial.ttf", 22)
    header_font = ImageFont.truetype("arial.ttf", 16)
    bold_font = ImageFont.truetype("arialbd.ttf", 15)
    text_font = ImageFont.truetype("arial.ttf", 15)
    small_font = ImageFont.truetype("arial.ttf", 13)
except IOError:
    title_font = ImageFont.load_default()
    header_font = ImageFont.load_default()
    bold_font = ImageFont.load_default()
    text_font = ImageFont.load_default()
    small_font = ImageFont.load_default()

# Data for 9 chunks
qp_data = [
    {
        "chunk_id": "01",
        "title": "PART 1: DATA COMMUNICATION FOUNDATIONS (For Pages 01-05)",
        "module": "MODULE 1",
        "questions": [
            ("1. a)", "What is Data Communication? Explain the five essential components", "of data communication with a neat diagram. (10 Marks)"),
            ("   b)", "Discuss the fundamental criteria required for an effective and", "efficient data communication network. (10 Marks)")
        ],
        "note": "Note: Answer Question 1(a) and 1(b). Evaluates content written on Pages 01-05."
    },
    {
        "chunk_id": "02",
        "title": "PART 2: TCP CONNECTION & PROTOCOLS (For Pages 06-10)",
        "module": "MODULE 4 & MODULE 1",
        "questions": [
            ("1. a)", "Describe the TCP Three-Way Handshake connection establishment", "and termination mechanism with a neat sequence diagram. (10 Marks)"),
            ("   b)", "Explain what happens if one of the handshake packets is lost or delayed.", "Also explain the 2MSL wait state during termination. (6 Marks)"),
            ("2. a)", "What is Flow Control and Congestion Control in TCP?", "Briefly introduce why both mechanisms are essential. (4 Marks)")
        ],
        "note": "Note: Answer all questions. Evaluates content written on Pages 06-10."
    },
    {
        "chunk_id": "03",
        "title": "PART 3: TCP CONGESTION & IPv4 HEADER (For Pages 11-15)",
        "module": "MODULE 4 & MODULE 3",
        "questions": [
            ("1. a)", "Explain the TCP Congestion Control mechanism including Slow Start,", "Congestion Window (cwnd), and threshold (SSTHRESH) calculation. (10 Marks)"),
            ("2. a)", "With a neat diagram, explain the IPv4 Datagram Header format and describe", "the significance of Version, HLEN, TTL, Checksum, and Protocol fields. (10 Marks)")
        ],
        "note": "Note: Answer all questions. Evaluates content written on Pages 11-15."
    },
    {
        "chunk_id": "04",
        "title": "PART 4: ROUTING & MULTIPLE ACCESS (For Pages 16-20)",
        "module": "MODULE 3 & MODULE 2",
        "questions": [
            ("1. a)", "Explain the Distance Vector Routing (DVR) algorithm with an illustrative", "network example calculating shortest path cost. (10 Marks)"),
            ("2. a)", "Explain Stop-and-Wait protocol. Differentiate between Pure ALOHA", "and Slotted ALOHA random access protocols. (10 Marks)")
        ],
        "note": "Note: Answer all questions. Evaluates content written on Pages 16-20."
    },
    {
        "chunk_id": "05",
        "title": "PART 5: NETWORK LAYER ROUTING & IPv6 (For Pages 21-25)",
        "module": "MODULE 3",
        "questions": [
            ("1. a)", "Differentiate between IPv4 and IPv6 header architectures. (10 Marks)", ""),
            ("2. a)", "Explain the working of Link State Routing (Dijkstra's Algorithm)", "with an example step-by-step table. (10 Marks)")
        ],
        "note": "Note: Test paper for Pages 21-25 (Practice & Unwritten Page Testing)."
    },
    {
        "chunk_id": "06",
        "title": "PART 6: TRANSPORT LAYER PROTOCOLS (For Pages 26-30)",
        "module": "MODULE 4",
        "questions": [
            ("1. a)", "Differentiate between TCP and UDP protocols in detail. (10 Marks)", ""),
            ("2. a)", "Explain Leaky Bucket and Token Bucket traffic shaping algorithms", "used for congestion control in networks. (10 Marks)")
        ],
        "note": "Note: Test paper for Pages 26-30 (Practice & Unwritten Page Testing)."
    },
    {
        "chunk_id": "07",
        "title": "PART 7: APPLICATION LAYER SERVICES (For Pages 31-35)",
        "module": "MODULE 5",
        "questions": [
            ("1. a)", "Explain Domain Name System (DNS) architecture and hierarchical resolution.", "(10 Marks)", ""),
            ("2. a)", "Discuss the working and architecture of HTTP and FTP protocols. (10 Marks)", "")
        ],
        "note": "Note: Test paper for Pages 31-35 (Practice & Unwritten Page Testing)."
    },
    {
        "chunk_id": "08",
        "title": "PART 8: DATA LINK LAYER & ERROR CONTROL (For Pages 36-40)",
        "module": "MODULE 2",
        "questions": [
            ("1. a)", "What is framing? Explain bit-stuffing and byte-stuffing methods. (10 Marks)", ""),
            ("2. a)", "Explain Cyclic Redundancy Check (CRC) error detection mechanism", "with a sample calculation. (10 Marks)")
        ],
        "note": "Note: Test paper for Pages 36-40 (Practice & Unwritten Page Testing)."
    },
    {
        "chunk_id": "09",
        "title": "PART 9: OSI REFERENCE MODEL & TOPOLOGIES (For Pages 41-44)",
        "module": "MODULE 1",
        "questions": [
            ("1. a)", "Explain the 7-layer OSI Reference Model with a neat diagram and explain", "the functionality of each layer. (10 Marks)"),
            ("2. a)", "Describe Star, Mesh, Bus, and Ring network topologies with their pros and cons.", "(10 Marks)", "")
        ],
        "note": "Note: Test paper for Pages 41-44 (Practice & Unwritten Page Testing)."
    }
]

def render_qp(data):
    width, height = 800, 1000
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Outer double border
    draw.rectangle([(15, 15), (785, 985)], outline=(0, 0, 0), width=2)
    draw.rectangle([(20, 20), (780, 980)], outline=(40, 40, 40), width=1)
    
    # USN Box (Top Right)
    draw.text((530, 35), "USN:", fill=(0, 0, 0), font=bold_font)
    draw.rectangle([(575, 30), (750, 58)], outline=(0, 0, 0), width=2)
    draw.text((585, 36), "1DB23CI034", fill=(0, 0, 0), font=bold_font)
    
    # Institution Header
    draw.text((120, 75), "VISVESVARAYA TECHNOLOGICAL UNIVERSITY, BELAGAVI", fill=(0, 0, 0), font=title_font)
    draw.text((250, 110), "Fifth Semester B.E. Degree Examination", fill=(0, 0, 0), font=header_font)
    draw.text((230, 135), "COMPUTER NETWORKS (Course Code: BCS502)", fill=(0, 0, 0), font=bold_font)
    
    # Subtitle / Chunk Name
    draw.rectangle([(40, 170), (760, 202)], fill=(240, 245, 255), outline=(100, 120, 160), width=1)
    draw.text((55, 178), data["title"], fill=(20, 40, 120), font=bold_font)
    
    # Exam Meta
    draw.text((45, 220), "Time: 1 hr", fill=(0, 0, 0), font=text_font)
    draw.text((640, 220), "Max. Marks: 20", fill=(0, 0, 0), font=text_font)
    draw.line([(40, 245), (760, 245)], fill=(0, 0, 0), width=2)
    
    # Note
    draw.text((45, 255), data["note"], fill=(70, 70, 70), font=small_font)
    draw.line([(40, 280), (760, 280)], fill=(180, 180, 180), width=1)
    
    # Module Header
    bbox = draw.textbbox((0, 0), data["module"], font=bold_font)
    mod_w = bbox[2] - bbox[0]
    cx = 400
    draw.rectangle([(cx - mod_w // 2 - 15, 300), (cx + mod_w // 2 + 15, 330)], fill=(230, 230, 230), outline=(0, 0, 0), width=1)
    draw.text((cx - mod_w // 2, 307), data["module"], fill=(0, 0, 0), font=bold_font)
    
    # Questions
    y = 360
    for item in data["questions"]:
        q_no, line1 = item[0], item[1]
        line2 = item[2] if len(item) > 2 else ""
        draw.text((45, y), q_no, fill=(0, 0, 0), font=bold_font)
        draw.text((105, y), line1, fill=(0, 0, 0), font=text_font)
        y += 28
        if line2.strip():
            draw.text((105, y), line2, fill=(0, 0, 0), font=text_font)
            y += 28
        y += 20
    
    # Footer Instructions
    draw.line([(40, 880), (760, 880)], fill=(180, 180, 180), width=1)
    draw.text((45, 895), "Instructions for Testing:", fill=(0, 0, 0), font=bold_font)
    draw.text((45, 920), "1. Select 'Computer Networks (CN)' in the Evaluator dropdown.", fill=(60, 60, 60), font=small_font)
    draw.text((45, 940), f"2. Upload this Question Paper along with Booklet_pages_{data['chunk_id']}_*.pdf.", fill=(60, 60, 60), font=small_font)
    draw.text((45, 960), "3. Click 'Evaluate Answer Script' to test automated grading and OCR accuracy.", fill=(60, 60, 60), font=small_font)
    
    # Save PNG and PDF
    base_name = f"QP_Chunk_{data['chunk_id']}"
    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    
    img.save(png_path)
    img.convert('RGB').save(pdf_path)
    print(f"Generated {base_name}.png and {base_name}.pdf")

for data in qp_data:
    render_qp(data)

# Also generate a Master QP covering the entire booklet
master_data = {
    "chunk_id": "Master_All_Questions",
    "title": "COMPREHENSIVE QUESTION PAPER (Covers All Answered Questions)",
    "module": "FULL BOOKLET (MODULES 1 to 4)",
    "questions": [
        ("Q1 a)", "Explain Data Communication and its five essential components with diagram. (10 Marks)", ""),
        ("Q1 b)", "State the characteristics and criteria for an effective network. (10 Marks)", ""),
        ("Q3 b)", "Explain Stop-and-Wait, Pure ALOHA, and Slotted ALOHA protocols. (10 Marks)", ""),
        ("Q6 a)", "Explain Distance Vector Routing (DVR) algorithm with an illustrative example. (10 Marks)", ""),
        ("Q6 b)", "Describe the IPv4 Datagram Header format with a neat field diagram. (10 Marks)", ""),
        ("Q8 a)", "Describe TCP Three-Way Handshake connection establishment and termination. (10 Marks)", ""),
        ("Q8 b)", "Explain Flow Control (Sliding Window) and Congestion Control in TCP. (10 Marks)", "")
    ],
    "note": "Master Question Paper for evaluating the entire 44-page booklet or multi-question tests."
}

def render_master_qp(data):
    width, height = 800, 1150
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([(15, 15), (785, 1135)], outline=(0, 0, 0), width=2)
    draw.rectangle([(20, 20), (780, 1130)], outline=(40, 40, 40), width=1)
    
    draw.text((530, 35), "USN:", fill=(0, 0, 0), font=bold_font)
    draw.rectangle([(575, 30), (750, 58)], outline=(0, 0, 0), width=2)
    draw.text((585, 36), "1DB23CI034", fill=(0, 0, 0), font=bold_font)
    
    draw.text((120, 75), "VISVESVARAYA TECHNOLOGICAL UNIVERSITY, BELAGAVI", fill=(0, 0, 0), font=title_font)
    draw.text((250, 110), "Fifth Semester B.E. Degree Examination", fill=(0, 0, 0), font=header_font)
    draw.text((230, 135), "COMPUTER NETWORKS (Course Code: BCS502)", fill=(0, 0, 0), font=bold_font)
    
    draw.rectangle([(40, 170), (760, 202)], fill=(240, 245, 255), outline=(100, 120, 160), width=1)
    draw.text((55, 178), data["title"], fill=(20, 40, 120), font=bold_font)
    
    draw.text((45, 220), "Time: 3 hrs", fill=(0, 0, 0), font=text_font)
    draw.text((630, 220), "Max. Marks: 70", fill=(0, 0, 0), font=text_font)
    draw.line([(40, 245), (760, 245)], fill=(0, 0, 0), width=2)
    
    draw.text((45, 255), data["note"], fill=(70, 70, 70), font=small_font)
    draw.line([(40, 280), (760, 280)], fill=(180, 180, 180), width=1)
    
    y = 300
    for q_no, line1, line2 in data["questions"]:
        draw.text((45, y), q_no, fill=(0, 0, 0), font=bold_font)
        draw.text((115, y), line1, fill=(0, 0, 0), font=text_font)
        y += 26
        if line2.strip():
            draw.text((115, y), line2, fill=(0, 0, 0), font=text_font)
            y += 26
        y += 18
    
    draw.line([(40, 1050), (760, 1050)], fill=(180, 180, 180), width=1)
    draw.text((45, 1065), "Instructions for Testing Full Booklet:", fill=(0, 0, 0), font=bold_font)
    draw.text((45, 1085), "Upload this Master QP along with the original 44-page booklet or any multi-page subset.", fill=(60, 60, 60), font=small_font)
    draw.text((45, 1105), "Ensure subject is set to 'Computer Networks (CN)' for comprehensive grading.", fill=(60, 60, 60), font=small_font)
    
    base_name = f"QP_{data['chunk_id']}"
    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    
    img.save(png_path)
    img.convert('RGB').save(pdf_path)
    print(f"Generated {base_name}.png and {base_name}.pdf")

render_master_qp(master_data)
print("All Question Papers generated successfully!")
