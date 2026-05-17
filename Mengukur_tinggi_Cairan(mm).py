import cv2
import argparse
import math 
from ultralytics import YOLO

def detect_video(weights, source, output, conf=0.25, device="CPU"):
    # --- KONFIGURASI KALIBRASI ---
    REF_PIXEL_HEIGHT = 900.0  # Tinggi referensi dalam piksel (misal: botol full)
    REF_REAL_HEIGHT  = 4.5    # Tinggi asli dunia nyata dalam mm (atau satuan lain)
    
    # Hitung faktor konversi (mm per piksel)
    MM_PER_PIXEL = REF_REAL_HEIGHT / REF_PIXEL_HEIGHT
    
    print(f"🚀 Loading model: {weights}")
    print(f"🖥️  Target Device: {device}")
    print(f"📏 Kalibrasi: 1 px = {MM_PER_PIXEL:.5f} mm") # Debugging info

    # Load YOLO model
    # Tambahkan task="detect" agar aman saat load model OpenVINO
    try:
        model = YOLO(weights, task="detect")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # PENTING UNTUK OPENVINO: 
    # Jangan gunakan model.to(device). OpenVINO menangani device saat inference.
    # model.to(device) <--- DIHAPUS

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Gagal membuka video: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output setup
    try:
        # Codec avc1 biasanya lebih bagus untuk web/player modern
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output, fourcc, fps, (width, height_video))
    except:
        # Fallback ke mp4v jika avc1 gagal
        print("⚠️ Codec avc1 tidak tersedia, menggunakan mp4v.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, (width, height_video))

    print(f"🎥 Processing: {source}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        # device='CPU', 'GPU' (Intel), atau 'AUTO'
        results = model.predict(frame, conf=conf, device=device, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                _, _, _, h_px = box.xywh[0].tolist() # h_px adalah tinggi dalam piksel

                # --- BAGIAN HITUNG MATEMATIKA ---
                # Hitung tinggi real (mm)
                h_mm = h_px * MM_PER_PIXEL
                
                # Format string untuk ditampilkan (2 angka desimal)
                text_info = f"{h_mm:.2f} mm"

                # Posisi text (di dalam kotak, turun 40px dari atas)
                text_x = int(x1) + 5
                text_y = int(y1) + 40

                # Gambar text dengan outline agar terbaca
                # 1. Outline Hitam (Tebal 4)
                cv2.putText(annotated_frame, text_info, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                # 2. Text Putih (Tebal 2)
                cv2.putText(annotated_frame, text_info, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(annotated_frame)
        
        # Preview window (50% size)
        preview = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
        cv2.imshow("YOLO OpenVINO Detection", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Selesai! Output tersimpan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated help text
    parser.add_argument("--weights", type=str, required=True, help="Path model (folder OpenVINO, file .xml, atau .pt)")
    parser.add_argument("--source", type=str, required=True, help="Video input (.mp4)")
    parser.add_argument("--output", type=str, default="result_mm.mp4", help="Video output")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    # Updated default device to CPU
    parser.add_argument("--device", type=str, default="CPU", help="Device: 'CPU', 'GPU' (Intel), 'AUTO'")
    
    args = parser.parse_args()
    detect_video(args.weights, args.source, args.output, args.conf, args.device)