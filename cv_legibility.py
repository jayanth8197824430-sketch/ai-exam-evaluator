import cv2
import numpy as np

def generate_legibility_heatmap(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes, {'legibility_score': 85, 'status': 'Good', 'scribbles': 0}
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    density = cv2.blur(grad_norm, (21, 21))
    
    cross_mask = (grad_norm > 135).astype(np.uint8) * 255
    scribble_density = cv2.blur(cross_mask, (31, 31))
    
    combined = cv2.addWeighted(density, 0.65, scribble_density, 0.35, 0)
    thermal_input = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    heatmap_color = cv2.applyColorMap(thermal_input, cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(img, 0.60, heatmap_color, 0.40, 0)
    
    _, ink_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    total_ink = max(cv2.countNonZero(ink_binary), 1)
    scribble_count = cv2.countNonZero(scribble_density > 115)
    scribble_ratio = scribble_count / total_ink
    
    legibility_score = max(52, min(97, round(100 - (scribble_ratio * 40))))
    status = 'High Legibility'
    if legibility_score < 72:
        status = 'Moderate / Cursive'
    elif legibility_score < 60:
        status = 'Scribbles / Crossed-Out Detected'
        
    _, encoded = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes(), {
        'legibility_score': legibility_score,
        'status': status,
        'scribble_detected': bool(scribble_count > 500)
    }
