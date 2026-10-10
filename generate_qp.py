from PIL import Image, ImageDraw, ImageFont

# Create a white A4-like canvas
width, height = 800, 1000
img = Image.new('RGB', (width, height), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Try to load a font, otherwise use default
try:
    title_font = ImageFont.truetype("arial.ttf", 24)
    header_font = ImageFont.truetype("arial.ttf", 18)
    text_font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    title_font = ImageFont.load_default()
    header_font = ImageFont.load_default()
    text_font = ImageFont.load_default()

# Header
draw.text((350, 40), "USN:", fill=(0,0,0), font=header_font)
draw.rectangle([(400, 35), (550, 60)], outline=(0,0,0), width=2)

draw.text((150, 100), "VISVESVARAYA TECHNOLOGICAL UNIVERSITY, BELAGAVI", fill=(0,0,0), font=title_font)
draw.text((250, 140), "Fifth Semester B.E. Degree Examination", fill=(0,0,0), font=header_font)
draw.text((320, 170), "COMPUTER NETWORKS (BCS502)", fill=(0,0,0), font=title_font)

draw.text((50, 220), "Time: 3 hrs.", fill=(0,0,0), font=text_font)
draw.text((650, 220), "Max. Marks: 100", fill=(0,0,0), font=text_font)
draw.line([(50, 240), (750, 240)], fill=(0,0,0), width=2)

draw.text((50, 260), "Note: Answer any FIVE full questions, choosing ONE full question from each module.", fill=(0,0,0), font=text_font)

# Questions
questions = [
    "MODULE - 1",
    "1. a) Explain the OSI Reference Model with a neat diagram. (10 Marks)",
    "   b) Differentiate between TCP and UDP protocols. (10 Marks)",
    "",
    "MODULE - 2",
    "2. a) What is framing? Explain any two framing methods used in Data Link Layer. (10 Marks)",
    "   b) Explain the CSMA/CD protocol in detail. (10 Marks)",
    "",
    "MODULE - 3",
    "3. a) Explain the Distance Vector Routing algorithm with an example. (12 Marks)",
    "   b) Differentiate between IPv4 and IPv6 header formats. (8 Marks)",
    "",
    "MODULE - 4",
    "4. a) Describe the TCP three-way handshake connection establishment. (10 Marks)",
    "   b) Explain the Leaky Bucket algorithm for congestion control. (10 Marks)",
    "",
    "MODULE - 5",
    "5. a) Explain the architecture of Domain Name System (DNS). (10 Marks)",
    "   b) Discuss the working of HTTP and FTP protocols. (10 Marks)"
]

y_pos = 320
for line in questions:
    if "MODULE" in line:
        draw.text((350, y_pos), line, fill=(0,0,0), font=header_font)
        y_pos += 40
    else:
        draw.text((60, y_pos), line, fill=(0,0,0), font=text_font)
        y_pos += 30

img.save('Generated_BCS502_Question_Paper.png')
print('Generated successfully!')
