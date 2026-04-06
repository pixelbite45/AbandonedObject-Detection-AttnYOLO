import time
import os
import sqlite3
import json
import cv2  # Added to check file integrity
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from attention import get_prediction 

# --- CONFIGURATION ---
FOLDER_TO_WATCH = "abondant_objects_clips"
MODEL_PATH = "attemton_model.pt"
JSON_FILE = "data.json"
# ---------------------

def is_file_ready(file_path, retries=5, delay=2):
    """
    Checks if the video file is complete and readable by OpenCV.
    """
    for i in range(retries):
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return True
        print(f"File not ready, retrying {i+1}/{retries}...")
        time.sleep(delay)
    return False

def push_to_json(filename):
    try:
        conn = sqlite3.connect("alerts.db")
        cursor = conn.cursor()
        # Search for the filename in the clip_url column
        cursor.execute(
            "SELECT camera_id, location, timestamp FROM alerts WHERE clip_url LIKE ? ORDER BY timestamp DESC LIMIT 1", 
            ('%' + filename + '%',)
        )
        row = cursor.fetchone()
        conn.close()

        cam_id, loc, ts = row if row else ("Unknown", "Unknown", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        new_entry = {
            "camera_no": cam_id,
            "location": loc,
            "timestamp": ts,
            "video_link": filename
        }

        data_list = []
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r') as f:
                try: data_list = json.load(f)
                except: data_list = []

        data_list.append(new_entry)
        with open(JSON_FILE, 'w') as f:
            json.dump(data_list, f, indent=2)
            
    except Exception as e:
        print(f"[JSON ERROR] {e}")

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".mp4"):
            file_path = event.src_path
            filename = os.path.basename(file_path)
            
            print(f"\n[NEW FILE] Detected: {filename}")
            
            # Use the integrity check instead of a blind sleep
            if is_file_ready(file_path):
                try:
                    result = get_prediction(file_path, MODEL_PATH) #
                    
                    if result == 1:
                        print(f"[KEEP] {filename} identified as ABANDONED.")
                        push_to_json(filename)
                    else:
                        print(f"[DELETE] {filename} identified as NORMAL. Removing...")
                        os.remove(file_path)
                except Exception as e:
                    print(f"[ERROR] Could not process {filename}: {e}")
            else:
                print(f"[SKIP] File {filename} remained unreadable (moov atom missing).")
            
            print("-" * 40)

if __name__ == "__main__":
    if not os.path.exists(FOLDER_TO_WATCH):
        os.makedirs(FOLDER_TO_WATCH)

    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, FOLDER_TO_WATCH, recursive=False)
    
    print(f"Monitoring folder: {os.path.abspath(FOLDER_TO_WATCH)}")
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()