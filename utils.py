import cv2
import numpy as np

VIRTUAL_LINE_Y = 400 

def crossing_virtual_line(cx, cy, prev_cy=None, direction='down'):
    if prev_cy is None:
        return False
    if direction == 'down':
        return prev_cy < VIRTUAL_LINE_Y <= cy
    else:
        return prev_cy > VIRTUAL_LINE_Y >= cy

def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def highlight_idle_vehicle(frame, ltrb, idle_duration=None, color=(0, 0, 255)):
    x1, y1, x2, y2 = map(int, ltrb)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, "IDLE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if idle_duration is not None:
        cv2.putText(
            frame,
            f"Idle: {idle_duration:.1f}s",
            (x1, y2 + 20),  
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

def draw_path(frame, path_points, color=(0,255,255), thickness=2):
    for i in range(1, len(path_points)):
        cv2.line(frame, path_points[i-1], path_points[i], color, thickness)

# deteksi warna dominan dari ROI
def get_dominant_color(roi):
    """
    Mengembalikan warna dominan HSV dari ROI.
    """
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return (0, 0, 0)

    img_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = img_hsv[:,:,2] > 60
    pixels = img_hsv[mask]

    if pixels.shape[0] == 0:
        return (0, 0, 0)

    # Histogram sederhana pakai bincount
    h = pixels[:,0]
    s = pixels[:,1]
    v = pixels[:,2]
    dom_h = int(np.bincount(h).argmax())
    dom_s = int(np.bincount(s).argmax())
    dom_v = int(np.bincount(v).argmax())
    return (dom_h, dom_s, dom_v)

def map_color(hsv_color):
    h, s, v = hsv_color

    if v < 60:
        return 'Black'
    if s < 30 and v > 200:
        return 'White'
    if s < 40 and 60 <= v <= 200:
        return 'Gray'

    # Warna dominan (chromatic colors)
    if s >= 40 and v >= 70:
        if h <= 10 or h >= 170:
            return 'Red'
        elif 11 <= h <= 25:
            return 'Orange'
        elif 26 <= h <= 34:
            return 'Yellow'
        elif 35 <= h <= 85:
            return 'Green'
        elif 86 <= h <= 125:
            return 'Blue'
        elif 126 <= h <= 169:
            return 'Purple'

    # Warna tidak dikenali
    return 'Other'
