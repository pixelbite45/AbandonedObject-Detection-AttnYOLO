# import for requirments
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time

# --- CONFIGURATION ---
MODEL_PATH = "yolov8n.pt"
VIDEO_STREAM_URL = "http://192.168.1.6:8080/video"

PERSON_CLASS_ID = 0
ITEM_CLASS_NAMES = ["backpack", "handbag", "suitcase", "laptop", "cell phone"]

# --- TUNABLE PARAMETERS ---
# This value is now set based on your log data.
ASSOCIATION_THRESHOLD = 850
# The time in seconds an item must be left alone to trigger an alert.
ALERT_PATIENCE_SECONDS = 5 # set for 5 sec interval if a person leave his object
RESIZE_FACTOR = 0.6

# --- HELPER FUNCTION ---
#equcladian distance 
def calculate_distance(p1: sv.Point, p2: sv.Point) -> float:
    """Calculates the Euclidean distance between two sv.Point objects."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- MAIN LOGIC ---
def main():
    model = YOLO(MODEL_PATH)
    item_class_ids = [k for k, v in model.model.names.items() if v in ITEM_CLASS_NAMES]

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    person_states = defaultdict(lambda: {"items": set()})
    tracked_items_info = {}
    
    
    # This now stores the timestamp when an item became an orphan
    orphan_candidates = {}

    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {VIDEO_STREAM_URL}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = np.array([cls in item_class_ids + [PERSON_CLASS_ID] for cls in detections.class_id])
        detections = detections[mask]
        tracked_detections = tracker.update_with_detections(detections)

        person_mask = tracked_detections.class_id == PERSON_CLASS_ID
        persons_detections = tracked_detections[person_mask]
        item_mask = np.isin(tracked_detections.class_id, item_class_ids)
        items_detections = tracked_detections[item_mask]

        for tracker_id, class_id in zip(items_detections.tracker_id, items_detections.class_id):
            tracked_items_info[tracker_id] = model.model.names[class_id]

        person_box_map = {tid: box for tid, box in zip(persons_detections.tracker_id, persons_detections.xyxy)}
        item_box_map = {tid: box for tid, box in zip(items_detections.tracker_id, items_detections.xyxy)}

        # --- LOGIC ---
        current_tracked_person_ids = set(persons_detections.tracker_id)
        current_tracked_item_ids = set(items_detections.tracker_id)

        # 1. Update associations
        for person_id, person_box in person_box_map.items():
            person_center = sv.Point(x=(person_box[0] + person_box[2]) / 2, y=(person_box[1] + person_box[3]) / 2)
            for item_id, item_box in item_box_map.items():
                item_center = sv.Point(x=(item_box[0] + item_box[2]) / 2, y=(item_box[1] + item_box[3]) / 2)
                if calculate_distance(person_center, item_center) < ASSOCIATION_THRESHOLD:
                    person_states[person_id]["items"].add(item_id)
                    # If a person claims an item, it's no longer an orphan
                    if item_id in orphan_candidates:
                        del orphan_candidates[item_id]

        # 2. Handle disappearing people to create orphan candidates
        for person_id in list(person_states.keys()):
            if person_id not in current_tracked_person_ids:
                for item_id in person_states[person_id]["items"]:
                    if item_id in current_tracked_item_ids and item_id not in orphan_candidates:
                        # Item is now an orphan, record the time
                        orphan_candidates[item_id] = time.time()
                del person_states[person_id]

        # 3. Process orphan candidates and trigger alerts
        for item_id in list(orphan_candidates.keys()):
            if item_id in current_tracked_item_ids:
                time_since_orphaned = time.time() - orphan_candidates[item_id]
                if time_since_orphaned > ALERT_PATIENCE_SECONDS:
                    item_name = tracked_items_info.get(item_id, "an item")
                    print(f"🚨 ALERT: An item '{item_name}' (ID: {item_id}) has been left unattended!")
                    del orphan_candidates[item_id] # Reset to avoid repeated alerts
            else:
                # The item also disappeared, so remove it
                del orphan_candidates[item_id]

        # --- VISUALIZATION ---
        labels = [f"#{tracker_id} {model.model.names[class_id]}" for tracker_id, class_id in zip(tracked_detections.tracker_id, tracked_detections.class_id)]
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        cv2.imshow("Abandoned Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()