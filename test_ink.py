import fitz
import cv2
import numpy as np

doc = fitz.open('1DB23CI034_BCS502_Booklet (2).pdf')
valid_pages = []

for i in range(len(doc)):
    pix = doc[i].get_pixmap(matrix=fitz.Matrix(1, 1))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif pix.n == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    h, w = gray.shape
    # Crop margins (VTU booklets have heavy margins, we only care about the writing area)
    # usually top 15%, bottom 10%, left 15%, right 10%
    center = gray[int(h*0.15):int(h*0.9), int(w*0.15):int(w*0.9)]
    
    # Adaptive threshold or simple threshold
    _, thresh = cv2.threshold(center, 180, 255, cv2.THRESH_BINARY_INV)
    ink_ratio = np.sum(thresh == 255) / thresh.size
    
    if ink_ratio > 0.005:  # 0.5% ink
        valid_pages.append((i, round(ink_ratio, 4)))

print(f'Total pages: {len(doc)}')
print(f'Pages with writing: {valid_pages}')
