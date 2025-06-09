import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, Counter

from utils import crossing_virtual_line, dist, highlight_idle_vehicle, draw_path, VIRTUAL_LINE_Y, get_dominant_color, map_color

model = YOLO("yolov8m.pt")
tracker = DeepSort()

vehicle_count = defaultdict(int)
vehicle_idle = {}
vehicle_paths = defaultdict(list)
vehicle_prev_pos = {}
vehicle_counted = {}
vehicle_last_seen = {}
vehicle_last_speed = {}
vehicle_color = {}              
vehicle_color_count = Counter() 
idle_threshold = 3  
COUNTED_MEMORY = 30 

cap = cv2.VideoCapture('traffic_video.mp4')
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_number = 0

PIXEL_TO_METER = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label in ['car', 'truck', 'bus']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2-x1, y2-y1], float(box.conf[0]), label))

    tracks = tracker.update_tracks(detections, frame=frame)
    current_track_ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        current_track_ids.add(track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = [int(val) for val in ltrb]
        label = track.det_class

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        vehicle_paths[track_id].append((cx, cy))

        h = y2 - y1
        w = x2 - x1
        crop_y1 = y1 + int(0.2 * h)
        crop_y2 = y2 - int(0.2 * h)
        crop_x1 = x1 + int(0.2 * w)
        crop_x2 = x2 - int(0.2 * w)
        roi = frame[max(crop_y1,0):max(crop_y2,0), max(crop_x1,0):max(crop_x2,0)]
        if roi.size > 0 and track_id not in vehicle_color:
            dom_color = get_dominant_color(roi)
            mapped_color = map_color(dom_color)
            vehicle_color[track_id] = mapped_color
            vehicle_color_count[mapped_color] += 1

        # kecepatan kendaraan
        prev_pos = vehicle_prev_pos.get(track_id, (cx, cy))
        jarak_pixel = dist(prev_pos, (cx, cy))
        jarak_meter = jarak_pixel / PIXEL_TO_METER
        waktu_detik = 1 / frame_rate if frame_rate > 0 else 1/25
        kecepatan_mps = jarak_meter / waktu_detik
        kecepatan_kmph = kecepatan_mps * 3.6
        vehicle_last_speed[track_id] = kecepatan_kmph

        # menghitung kendaraan yang melewati garis 
        prev_cy = prev_pos[1]
        if (
            (track_id not in vehicle_counted or not vehicle_counted[track_id]) and
            crossing_virtual_line(cx, cy, prev_cy)
        ):
            vehicle_count[label] += 1
            vehicle_counted[track_id] = True
        vehicle_prev_pos[track_id] = (cx, cy)
        vehicle_last_seen[track_id] = frame_number

        # durasi idle kendaraan
        prev_idle_pos = vehicle_idle.get(track_id, {}).get('pos')
        prev_time = vehicle_idle.get(track_id, {}).get('time', frame_number)
        if prev_idle_pos and dist(prev_idle_pos, (cx, cy)) < 5:
            idle_duration = (frame_number - prev_time) / frame_rate
            if idle_duration > idle_threshold:
                highlight_idle_vehicle(frame, (x1, y1, x2, y2), idle_duration)
        else:
            vehicle_idle[track_id] = {'pos': (cx, cy), 'time': frame_number}

        # visualisasi kecepatan dan warna kendaraan
        color_label = vehicle_color.get(track_id, "Unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{label}-{track_id} | {vehicle_last_speed[track_id]:.1f} km/h | {color_label}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2
        )
        draw_path(frame, vehicle_paths[track_id])

    # Reset counted jika track_id hilang beberapa frame 
    for tid in list(vehicle_counted.keys()):
        if tid not in current_track_ids:
            last_seen = vehicle_last_seen.get(tid, frame_number)
            if frame_number - last_seen > COUNTED_MEMORY:
                del vehicle_counted[tid]
                del vehicle_prev_pos[tid]
                del vehicle_last_seen[tid]
                if tid in vehicle_paths:
                    del vehicle_paths[tid]
                if tid in vehicle_last_speed:
                    del vehicle_last_speed[tid]
                if tid in vehicle_idle:
                    del vehicle_idle[tid]
                if tid in vehicle_color:
                    vehicle_color_count[vehicle_color[tid]] -= 1
                    if vehicle_color_count[vehicle_color[tid]] <= 0:
                        del vehicle_color_count[vehicle_color[tid]]
                    del vehicle_color[tid]

    # garis 
    cv2.line(frame, (0, VIRTUAL_LINE_Y), (frame.shape[1], VIRTUAL_LINE_Y), (0, 255, 255), 2)

    # Show counters kendaraan
    y = 30
    for k, v in vehicle_count.items():
        cv2.putText(frame, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 30

    # counters warna kendaraan
    y_color = y + 10
    for color, v in vehicle_color_count.items():
        cv2.putText(frame, f"{color}: {v}", (10, y_color), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        y_color += 30

    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_number += 1

cap.release()
cv2.destroyAllWindows()