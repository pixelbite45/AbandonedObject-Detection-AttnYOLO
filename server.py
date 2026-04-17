import os
import shutil
import sqlite3
import time
from datetime import datetime

# Configuration
DB_PATH = "alerts.db"
SOURCE_DIR = "video_chunks"
DEST_DIR = "abondant_objects_clips"

def is_file_stable(filepath):
    """Checks if file size is no longer increasing, meaning writing is done."""
    try:
        first_size = os.path.getsize(filepath)
        time.sleep(1) # Wait a second
        second_size = os.path.getsize(filepath)
        return first_size == second_size and first_size > 0
    except:
        return False

def process_alerts():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp FROM alerts WHERE is_checked = 0")
        alerts = cursor.fetchall()
        
        if not alerts:
            conn.close()
            return

        video_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".mp4")]
        
        for alert_id, alert_time_str in alerts:
            alert_dt = datetime.strptime(alert_time_str, "%Y-%m-%d %H:%M:%S")
            best_match = None
            min_diff = float('inf')

            for video_file in video_files:
                try:
                    parts = video_file.replace(".mp4", "").split('_')
                    video_dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
                    diff = (alert_dt - video_dt).total_seconds()
                    
                    if 0 <= diff <= 80: 
                        if diff < min_diff:
                            min_diff = diff
                            best_match = video_file
                except:
                    continue

            if best_match:
                src_path = os.path.join(SOURCE_DIR, best_match)
                
                # CRITICAL: Check if the file is ready to be moved
                if is_file_stable(src_path):
                    if not os.path.exists(DEST_DIR): os.makedirs(DEST_DIR)
                    shutil.copy2(src_path, os.path.join(DEST_DIR, best_match))
                    
                    cursor.execute("UPDATE alerts SET is_checked = 1 WHERE id = ?", (alert_id,))
                    print(f"✅ Alert {alert_id} matched & moved: {best_match}")
                else:
                    print(f"⏳ File {best_match} is still being written. Waiting...")

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    while True:
        process_alerts()
        time.sleep(5)