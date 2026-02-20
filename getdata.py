import cv2
import pandas as pd
import easyocr
import numpy as np
import requests
import os
import time

# 1. Setup OCR 
reader = easyocr.Reader(['en', 'ja'], gpu=False)

# 2. Configuration: Mapping the 4 unique table layouts
image_configs = {
    "group_1": {"suffix": "-1-670.png", "num_cols": 11, "margin": 0, "rides": ["Time", "Frozen", "Rapunzel", "PeterPan", "Tinker", "Soarin", "ToT", "ToyStory", "Indy", "Raging", "Center"]},
    "group_2": {"suffix": "-2-670.png", "num_cols": 8, "margin": 30, "rides": ["Time", "Steamer_Med", "Gondola", "Fortress", "TurtleTalk", "BigCity", "Steamer_Am", "Railway_Am"]},
    "group_3": {"suffix": "-3-670.png", "num_cols": 9, "margin": 0, "rides": ["Time", "Nemo", "Aquatopia", "Railway_PD", "Steamer_Lost", "MagicLamp", "Jasmine", "Sindbad", "Caravan"]},
    "group_4": {"suffix": "-4-670.png", "num_cols": 9, "margin": 0, "rides": ["Time", "Mermaid_Theater", "Ariel_Play", "FlyingFish", "Scuttle", "Jellyfish", "Blowfish", "Whirlpool", "20000Leagues"]}
}

def process_single_image(img_url, config, retries=3):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = None
    for attempt in range(retries):
        try:
            response = requests.get(img_url, headers=headers, timeout=30)
            response.raise_for_status() 
            break 
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return pd.DataFrame()

    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return pd.DataFrame()

    h, w, _ = img.shape
    header_h, footer_buffer, num_rows = 320, 51, 30
    row_height = (h - header_h - footer_buffer) / num_rows
    col_width = (w - (config["margin"] * 2)) / config["num_cols"]
    fixed_times = pd.date_range("07:15", "21:45", freq="30min").strftime("%H:%M").tolist()
    
    all_cells = []
    for r in range(num_rows):
        row_data = []
        row_data.append(fixed_times[r]) 
        y_s, y_e = int(header_h + (r * row_height)), int(header_h + ((r + 1) * row_height))
        
        for c in range(1, config["num_cols"]):
            x_s, x_e = int(config["margin"] + (c * col_width)), int(config["margin"] + ((c+1) * col_width))
            cell = img[y_s+2 : y_e-2, x_s+2 : x_e-2]
            
            # --- ATTEMPT 1: Your Standard Pass ---
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contrast = clahe.apply(gray)
            scaled = cv2.resize(contrast, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
            _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(thresh) < 127: thresh = cv2.bitwise_not(thresh)
            
            result = reader.readtext(thresh, detail=0, allowlist='0123456789-休止', paragraph=False)
            raw_val = result[0] if result else "-"

            # --- ATTEMPT 2: Your Repair Pass ---
            if raw_val == "-" or not any(char.isdigit() for char in raw_val):
                repair_kernel = np.ones((2,2), np.uint8)
                bold_thresh = cv2.erode(cv2.bitwise_not(thresh), repair_kernel, iterations=1)
                bold_thresh = cv2.bitwise_not(bold_thresh)
                result_retry = reader.readtext(bold_thresh, detail=0, allowlist='0123456789-休止', paragraph=False)
                if result_retry: raw_val = result_retry[0]

            # --- NEW ATTEMPT 3: Adaptive Gaussian Rescue (The 'Frozen' Saver) ---
            # Specialized for dark red cells where Otsu fails
            if raw_val == "-" or not any(char.isdigit() for char in raw_val):
                adapt_thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY_INV, 15, 8)
                # Clean noise while keeping digits thin to prevent '0' from filling in
                kernel = np.ones((2,2), np.uint8)
                cleaned = cv2.morphologyEx(adapt_thresh, cv2.MORPH_OPEN, kernel)
                
                result_final = reader.readtext(cleaned, detail=0, allowlist='0123456789-休止', paragraph=False)
                if result_final:
                    raw_val = result_final[0]

            clean_val = "CLOSED" if any(char in raw_val for char in ["休", "止"]) else ("".join(filter(str.isdigit, raw_val)) or "-")
            row_data.append(clean_val)
            
        all_cells.append(row_data)
    
    return pd.DataFrame(all_cells, columns=config["rides"])

# --- MAIN EXECUTION: DATE RANGE LOOP ---
start_date = "2025-09-09"
end_date = "2025-09-30"
skip_dates = ["20250801", "20250919", "20250929"]
master_file = "disney_sea_april_history.csv"

# Generate the list of dates to process
date_list = pd.date_range(start=start_date, end=end_date).strftime("%Y%m%d").tolist()

print(f"Starting multi-day extraction for {len(date_list)} days...")

for target_date in date_list:
    if target_date in skip_dates:
        print(f"\n>>> Skipping {target_date} (User requested skip)")
        continue
    
    day_parts = []
    print(f"\nProcessing Date: {target_date}")
    
    for key, config in image_configs.items():
        url = f"https://disneyreal.asumirai.info/realtime/images/sea-wait-{target_date}{config['suffix']}"
        print(f"  > Group: {key}...")
        
        try:
            df_part = process_single_image(url, config)
            if not df_part.empty:
                day_parts.append(df_part)
        except Exception as e:
            print(f"    ! Error processing {key} on {target_date}: {e}")

        # Pause to avoid rate-limiting
        time.sleep(5)

    if day_parts:
        # 1. Anchor all 4 tables on the "Time" column
        final_df = day_parts[0]
        for i in range(1, len(day_parts)):
            next_df = day_parts[i]
            final_df = pd.merge(final_df, next_df, on="Time", how="outer")

        # 2. Add Date column at the start
        if "Date" not in final_df.columns:
            final_df.insert(0, "Date", target_date)

        # 3. Whole-Day CLOSED Logic (Maintenance Broadcast)
        for ride in final_df.columns:
            if ride in ["Date", "Time"]: continue
            if "CLOSED" in final_df[ride].values:
                final_df[ride] = "CLOSED"

        # 4. Save/Append to CSV
        # header=not os.path.exists(master_file) ensures headers only at the top
        file_exists = os.path.exists(master_file)
        final_df.to_csv(master_file, mode='a', index=False, header=not file_exists)
        
        print(f"SUCCESS: Data for {target_date} appended to {master_file}")
    else:
        print(f"FAILED: No data extracted for {target_date}")

print("\n--- ALL DAYS COMPLETE ---")
