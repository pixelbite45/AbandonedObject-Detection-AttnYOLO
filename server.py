import os
import shutil
import sqlite3
import time
from datetime import datetime

# Configuration
DB_PATH = "alerts.db"
SOURCE_DIR = "video_chunks"
DEST_DIR = "abondant_objects_clips"

def process_alerts():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Select unchecked alerts
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
                    # Parse: chunk_20260327_215657.mp4
                    parts = video_file.replace(".mp4", "").split('_')
                    video_dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")

                    # Calculate seconds difference
                    diff = (alert_dt - video_dt).total_seconds()
                    
                    # FIX: We broaden the window to 20 seconds.
                    # This captures the 10s chunk + 10s of processing/patience delay.
                    if 0 <= diff <= 20: 
                        if diff < min_diff:
                            min_diff = diff
                            best_match = video_file
                except:
                    continue

            if best_match:
                if not os.path.exists(DEST_DIR): os.makedirs(DEST_DIR)
                shutil.copy2(os.path.join(SOURCE_DIR, best_match), os.path.join(DEST_DIR, best_match))
                
                cursor.execute("UPDATE alerts SET is_checked = 1 WHERE id = ?", (alert_id,))
                print(f"✅ Alert {alert_id} matched with {best_match} (Gap: {diff}s)")
            else:
                print(f"🔍 Still looking for a video starting 0-20s before {alert_time_str}...")

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    while True:
        process_alerts()
        time.sleep(5)