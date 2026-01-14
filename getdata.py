import cv2
import pandas as pd
import easyocr
import numpy as np
import requests
import re

# 1. Setup OCR (Using GPU for speed and accuracy)
reader = easyocr.Reader(['en', 'ja'], gpu=True)

# 2. Get the image from the link
img_url = "https://disneyreal.asumirai.info/realtime/images/sea-wait-20260112-1-670.png"
response = requests.get(img_url)
nparr = np.frombuffer(response.content, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

if img is None:
    print("Error: Could not load image from link.")
    exit()

h, w, _ = img.shape
print(f"Image Loaded: {w}x{h}")

# --- GRID DIMENSIONS ---
header_h = 320   
footer_h = 50    
num_cols = 11    
num_rows = 30    

col_width = w / num_cols
row_height = (h - header_h - footer_h) / num_rows
padding = 3  

all_rows = []
rides = ["Time", "Frozen", "Rapunzel", "PeterPan", "Tinker", "Soarin", "ToT", "ToyStory", "Indy", "Raging", "Center"]

print("Starting OCR extraction...")

for r in range(num_rows):
    current_row_data = []
    y_start = int(header_h + (r * row_height))
    y_end = int(y_start + row_height)
    
    for c in range(num_cols):
        x_start = int(c * col_width)
        x_end = int(x_start + col_width)
        
        # 3. Slice the individual cell
        cell = img[y_start + padding : y_end - padding, 
                   x_start + padding : x_end - padding]
        
        # 4. High-Contrast Pre-processing
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)
            
        # 5. Run OCR and INTEGRATED CLEANING
        result = reader.readtext(thresh, detail=0, allowlist='0123456789:-休止')
        raw_val = result[0] if result else "-"

        # --- DATA CLEANING LAYER ---
        if c == 0:  # TIME COLUMN CLEANING
            # Strip everything but digits
            clean_time = "".join(filter(str.isdigit, raw_val))
            # Fix common OCR jumps (e.g., 16345 -> 16:45)
            if len(clean_time) >= 4:
                # Take only the first two and last two digits to ignore OCR hallucinations
                clean_val = f"{clean_time[:2]}:{clean_time[-2:]}"
            elif len(clean_time) == 3:
                clean_val = f"0{clean_time[0]}:{clean_time[1:]}"
            else:
                clean_val = raw_val
        
        else:  # RIDE COLUMNS CLEANING
            if any(char in raw_val for char in ["休", "止"]):
                clean_val = "-"
            else:
                # Keep only digits for wait times to remove OCR artifacts
                digits = "".join(filter(str.isdigit, raw_val))
                clean_val = digits if digits else "-"

        current_row_data.append(clean_val)
    
    all_rows.append(current_row_data)
    print(f"Processed {current_row_data[0]}...") 

# 6. Create the Final Table
df = pd.DataFrame(all_rows, columns=rides)

# Save for your Route Optimization project
df.to_csv("disney_sea_history.csv", index=False)
print("\n--- PHASE 1 COMPLETE ---")
print("Data saved to 'disney_sea_history.csv'")
print(df.head())