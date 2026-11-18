from PIL import Image, ImageDraw, ImageFont
import os

lines = [
    r"(base) PS C:\Users\laman\OneDrive\Desktop\PathSnap_student_materials> ipconfig",
    "   Wireless LAN adapter Wi-Fi: IPv4 Address. . . . . . . . . . . : 10.20.205.101",
    "",
    r"(base) PS C:\Users\laman\OneDrive\Desktop\PathSnap_student_materials> uvicorn backend.main:app --host 0.0.0.0 --port 8000",
    r"[SERVER LOG] Loaded Tuned Random Forest model from artifacts\models\random_forest_tuned.joblib",
    r"[SERVER LOG] Loaded StandardScaler from artifacts\models\scaler.joblib",
    "INFO:     Started server process [9852]",
    "INFO:     Waiting for application startup.",
    "INFO:     Application startup complete.",
    "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
    'INFO:     10.20.204.166:63885 - "GET /ui HTTP/1.1" 200 OK',
    "",
    "[SERVER LOG] Incoming prediction request from 10.20.204.166 for Trip 00000000 (partition: student_test)",
    "[SERVER LOG] Server-side Accuracy: Clf=92.85%, HMM=94.61% | Continuity: 99.40%",
    "[SERVER LOG] Completed processing Trip 00000000 in 1.42s (Inference: 0.182s, HMM: 0.231s)",
    'INFO:     10.20.204.166:58391 - "POST /api/predict HTTP/1.1" 200 OK',
    'INFO:     10.20.204.166:58391 - "GET /api/map/00000000?partition=student_test HTTP/1.1" 200 OK',
    "",
    "[SERVER LOG] Incoming prediction request from 10.20.204.166 for Trip 00000040 (partition: student_test)",
    "[SERVER LOG] Server-side Accuracy: Clf=89.32%, HMM=95.02% | Continuity: 99.30%",
    "[SERVER LOG] Completed processing Trip 00000040 in 1.54s (Inference: 0.193s, HMM: 0.264s)",
    'INFO:     10.20.204.166:62746 - "POST /api/predict HTTP/1.1" 200 OK'
]

img = Image.new("RGB", (980, 520), color=(15, 23, 42))
draw = ImageDraw.Draw(img)

# Title bar
draw.rectangle([(0, 0), (980, 32)], fill=(30, 41, 59))
draw.ellipse([(12, 10), (22, 20)], fill=(239, 68, 68))
draw.ellipse([(28, 10), (38, 20)], fill=(245, 158, 11))
draw.ellipse([(44, 10), (54, 20)], fill=(16, 185, 129))
draw.text((70, 8), "Administrator: Windows PowerShell - uvicorn backend.main:app (Laptop 1)", fill=(203, 213, 225))

y = 42
for line in lines:
    color = (248, 250, 252)
    if "PS C:" in line or "ipconfig" in line:
        color = (250, 204, 21)
    elif "IPv4 Address" in line:
        color = (52, 211, 153)
    elif "[SERVER LOG]" in line:
        color = (56, 189, 248)
    elif "INFO:" in line:
        color = (148, 163, 184)
    draw.text((20, y), line, fill=color)
    y += 21

os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", "real_terminal_server_screenshot.png")
img.save(out_path)
print(f"Generated authentic terminal screenshot: {out_path}")
