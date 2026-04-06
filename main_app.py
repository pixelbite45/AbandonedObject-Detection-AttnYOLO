import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time
import os
from datetime import datetime
import sqlite3

CAMERA_ID = "CAM_01"
LOCATION = "Main Entrance"

# --- Configuration ---
MODEL_PATH = "yolo11n.pt"
VIDEO_STREAM_URL = r"C:\Users\rohan\Desktop\college\Innovative vii\test3.mp4"
ITEM_CLASS_NAMES = ["backpack", "handbag", "suitcase", "laptop", "cell phone"]
PERSON_CLASS_ID = 0
CHUNK_DURATION_SECONDS = 10 
VIDEO_OUTPUT_DIR = "video_chunks"
ASSOCIATION_THRESHOLD = 150
ALERT_PATIENCE_SECONDS = 5
RESIZE_FACTOR = 0.6

# --- Define your Border/Zone here ---
LINE_START = sv.Point(50, 400)
LINE_END = sv.Point(1230, 400)

# --- Utilities ---

def log_alert_to_db(item_name, person_id, camera_id, location, clip_url):
    try:
        conn = sqlite3.connect("alerts.db")
        cursor = conn.cursor()
        current_local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure there are exactly 7 '?' placeholders for the 7 columns
        query = """
            INSERT INTO alerts (timestamp, item_name, person_id, camera_id, location, clip_url, is_checked) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (current_local_time, item_name, person_id, camera_id, location, clip_url, 0))

        conn.commit()
        conn.close()
        print(f"Successfully logged alert: {item_name}")
    except Exception as e:
        print(f"Database Insertion Error: {e}")
def get_video_writer(frame, fps):
    if not os.path.exists(VIDEO_OUTPUT_DIR):
        os.makedirs(VIDEO_OUTPUT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(VIDEO_OUTPUT_DIR, f"chunk_{timestamp}.mp4")
    
    # Use 'mp4v' and ensure the writer opens correctly
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    
    return writer, filename

def main():
    model = YOLO(MODEL_PATH)
    item_class_ids = [k for k, v in model.model.names.items() if v in ITEM_CLASS_NAMES]
    
    tracker = sv.ByteTrack()
    # RESTORED: Line Zone and Annotators
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    person_states = defaultdict(lambda: {"items": set()})
    orphan_candidates = {} 

    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # --- Exact 10-second Chunk Logic ---
    frames_per_chunk = fps * CHUNK_DURATION_SECONDS
    frame_counter = 0
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Start new chunk if limit reached or first frame
        if out is None or frame_counter >= frames_per_chunk:
            if out:
                out.release()
            out, current_clip_path = get_video_writer(frame, fps)
            frame_counter = 0
        
        out.write(frame)
        frame_counter += 1

        # Detection and Tracking
        resized_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        results = model(resized_frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, item_class_ids + [PERSON_CLASS_ID])]
        tracked_detections = tracker.update_with_detections(detections)

        # Update Line Zone
        line_zone.trigger(detections=tracked_detections)

        persons = tracked_detections[tracked_detections.class_id == PERSON_CLASS_ID]
        items = tracked_detections[np.isin(tracked_detections.class_id, item_class_ids)]

        curr_person_ids = set(persons.tracker_id)
        curr_item_ids = set(items.tracker_id)

        # Association Logic
        for p_id, p_box in zip(persons.tracker_id, persons.xyxy):
            p_center = sv.Point((p_box[0]+p_box[2])/2, (p_box[1]+p_box[3])/2)
            for i_id, i_box in zip(items.tracker_id, items.xyxy):
                i_center = sv.Point((i_box[0]+i_box[2])/2, (i_box[1]+i_box[3])/2)
                dist = np.sqrt((p_center.x - i_center.x)**2 + (p_center.y - i_center.y)**2)
                if dist < ASSOCIATION_THRESHOLD:
                    person_states[p_id]["items"].add(i_id)
                    orphan_candidates.pop(i_id, None)

        # Abandonment check
        for p_id in list(person_states.keys()):
            if p_id not in curr_person_ids:
                owner_data = person_states.pop(p_id)
                for i_id in owner_data["items"]:
                    if i_id in curr_item_ids and i_id not in orphan_candidates:
                        orphan_candidates[i_id] = {"timestamp": time.time(), "person_id": p_id}

        # DB Alert Trigger
        for i_id in list(orphan_candidates.keys()):
            if i_id in curr_item_ids:
                if time.time() - orphan_candidates[i_id]["timestamp"] > ALERT_PATIENCE_SECONDS:
                    idx = np.where(items.tracker_id == i_id)[0][0]
                    i_name = model.model.names[items.class_id[idx]]
                    data = orphan_candidates.pop(i_id)
                    log_alert_to_db(i_name, data["person_id"],CAMERA_ID,LOCATION,current_clip_path)
                    print(f"🚨 ALERT: {i_name} logged to Database.")
            else:
                orphan_candidates.pop(i_id, None)

        # Visualization (Restored Borders)
        labels = [f"#{tid} {model.model.names[cid]}" for tid, cid in zip(tracked_detections.tracker_id, tracked_detections.class_id)]
        anno_frame = box_annotator.annotate(resized_frame.copy(), tracked_detections)
        anno_frame = label_annotator.annotate(anno_frame, tracked_detections, labels)
        anno_frame = line_annotator.annotate(frame=anno_frame, line_counter=line_zone)

        cv2.imshow("Abandoned Object Detection", anno_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    if out: out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()