"""
OMR Processing Module for NEET OMR Checker
Uses computer vision to detect filled bubbles in OMR sheets.
"""

import cv2
import numpy as np
from PIL import Image
import io
import os


def preprocess_image(image_input):
    """Convert uploaded file or PIL image to OpenCV format."""
    if isinstance(image_input, np.ndarray):
        img = image_input
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_input.seek(0)
    return img


def find_omr_region(img):
    """Find the main OMR answer area in the sheet."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = img.shape[:2]
    
    best_rect = None
    best_area = 0
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area > (w * h * 0.2) and cw > 100 and ch > 100:
            if area > best_area:
                best_area = area
                best_rect = (x, y, cw, ch)
    
    if best_rect:
        return best_rect
    
    return (0, 0, w, h)


def detect_bubbles_in_column(img, col_x, col_y, col_w, col_h, num_rows=50, num_options=4):
    """
    Detect filled bubbles in a single column of the OMR sheet.
    Returns list of detected answers (0=A, 1=B, 2=C, 3=D, -1=unattempted, -2=multiple)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_region = gray[col_y:col_y+col_h, col_x:col_x+col_w]
    
    row_h = col_h / num_rows
    bubble_w = col_w / num_options
    
    answers = []
    
    for row in range(num_rows):
        row_start = int(row * row_h)
        row_end = int((row + 1) * row_h)
        
        darknesses = []
        for opt in range(num_options):
            opt_start = int(opt * bubble_w)
            opt_end = int((opt + 1) * bubble_w)
            
            cell = col_region[row_start:row_end, opt_start:opt_end]
            
            if cell.size == 0:
                darknesses.append(255)
                continue
            
            _, cell_thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            filled_ratio = np.sum(cell_thresh > 0) / cell_thresh.size
            darknesses.append(filled_ratio)
        
        max_dark = max(darknesses)
        
        if max_dark < 0.08:
            answers.append(-1)
        else:
            filled = [i for i, d in enumerate(darknesses) if d > max_dark * 0.5 and d > 0.08]
            if len(filled) == 0:
                answers.append(-1)
            elif len(filled) == 1:
                answers.append(filled[0])
            else:
                answers.append(-2)
    
    return answers


def auto_detect_columns(img):
    """
    Automatically detect the 4 answer columns in the OMR sheet.
    Returns list of (x, y, w, h) for each column.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    row_sums = np.sum(thresh > 0, axis=0)
    
    top_margin = int(h * 0.35)
    bottom_margin = int(h * 0.98)
    left_margin = int(w * 0.05)
    right_margin = int(w * 0.95)
    
    col_w = (right_margin - left_margin) // 4
    
    columns = []
    for i in range(4):
        x = left_margin + i * col_w
        columns.append((x, top_margin, col_w, bottom_margin - top_margin))
    
    return columns


def process_omr_image(image_input, num_rows=50, num_options=4):
    """
    Main function to process an OMR image and extract answers.
    Returns dict with answers for each column.
    """
    img = preprocess_image(image_input)
    
    h, w = img.shape[:2]
    
    top_margin = int(h * 0.37)
    bottom_margin = int(h * 0.98)
    left_margin = int(w * 0.06)
    right_margin = int(w * 0.95)
    
    usable_w = right_margin - left_margin
    col_w = usable_w // 4
    
    results = {}
    
    for col_idx in range(4):
        col_x = left_margin + col_idx * col_w
        col_y = top_margin
        col_h = bottom_margin - top_margin
        
        answers = detect_bubbles_in_column(
            img, col_x, col_y, col_w, col_h,
            num_rows=num_rows, num_options=num_options
        )
        results[f"col_{col_idx + 1}"] = answers
    
    return results


def visualize_detection(image_input, detected_answers):
    """
    Draw bounding boxes on detected bubbles for visualization.
    Returns annotated image as PIL Image.
    """
    img = preprocess_image(image_input)
    h, w = img.shape[:2]
    
    vis_img = img.copy()
    
    top_margin = int(h * 0.37)
    bottom_margin = int(h * 0.98)
    left_margin = int(w * 0.06)
    right_margin = int(w * 0.95)
    
    usable_w = right_margin - left_margin
    col_w = usable_w // 4
    
    option_colors = [
        (0, 255, 0),
        (0, 128, 255),
        (0, 0, 255),
        (255, 0, 255)
    ]
    
    for col_idx in range(4):
        col_x = left_margin + col_idx * col_w
        col_y = top_margin
        col_h = bottom_margin - top_margin
        
        row_h = col_h / 50
        bubble_w = col_w / 4
        
        col_key = f"col_{col_idx + 1}"
        answers = detected_answers.get(col_key, [])
        
        for row_idx, ans in enumerate(answers):
            if ans >= 0:
                row_start = int(col_y + row_idx * row_h)
                row_end = int(col_y + (row_idx + 1) * row_h)
                opt_start = int(col_x + ans * bubble_w)
                opt_end = int(col_x + (ans + 1) * bubble_w)
                
                color = option_colors[ans % 4]
                cv2.rectangle(vis_img, (opt_start, row_start), (opt_end, row_end), color, 2)
                
            elif ans == -2:
                row_start = int(col_y + row_idx * row_h)
                row_end = int(col_y + (row_idx + 1) * row_h)
                cv2.rectangle(vis_img, (col_x, row_start), (col_x + col_w, row_end), (0, 165, 255), 2)
    
    rgb_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)
