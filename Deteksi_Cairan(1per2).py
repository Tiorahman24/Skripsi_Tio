import cv2
import argparse
from ultralytics import YOLO

def detect_video(weights, source, output, conf=0.25, device="CPU"):
    # ==========================================
    # 1. SETUP NILAI LITERATUR (VOLUME ASLI)
    # ==========================================
    # Ubah nilai ini sesuai hasil timbangan benda yang sedang dites
    BERAT_ISI_GRAM     = 34.0 #input dari sensor   
    BERAT_KOSONG_GRAM  = 14.0   
    MASSA_JENIS        = 1.1485 #3.3 

    # Hitung Volume Asli (Target Kebenaran)
    massa_cairan = BERAT_ISI_GRAM - BERAT_KOSONG_GRAM
    VOLUME_ASLI_ML = massa_cairan / MASSA_JENIS  
    
    # ==========================================
    # 2. SETUP KALIBRASI YOLO (PIKSEL KE MM & ML)
    # ==========================================
    REF_PIXEL_HEIGHT   = 920.0  # Tinggi piksel saat full
    
    # Kalibrasi Volume (ML)
    REF_FULL_VOLUME_ML = 30.0   
    ML_PER_PIXEL       = REF_FULL_VOLUME_ML / REF_PIXEL_HEIGHT
    
    # Kalibrasi Tinggi (MM)
    REF_REAL_HEIGHT_MM = 45.5    
    MM_PER_PIXEL       = REF_REAL_HEIGHT_MM / REF_PIXEL_HEIGHT

    print(f"🚀 Loading OpenVINO model: {weights}")
    print(f"🖥️  Device Target: {device}")
    print(f"🧪 Target Volume Asli (Literatur): {VOLUME_ASLI_ML:.2f} ml")

    # Load Model (Support OpenVINO XML or Folder)
    try:
        model = YOLO(weights, task="detect")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # NOTE: model.to(device) REMOVED because OpenVINO handles device in predict()

    # Buka Video/Kamera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Gagal membuka video: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video Writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output, fourcc, fps, (width, height_video))
    except:
        print("⚠️ Codec avc1 unavailable, using mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, (width, height_video))

    print(f"🎥 Processing: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run Inference
        # device='CPU' or 'GPU' (Intel) works for OpenVINO
        results = model.predict(frame, conf=conf, device=device, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                _, _, _, h_px = box.xywh[0].tolist()

                # --- PERHITUNGAN ---
                
                # 1. Tinggi (T) dalam mm
                tinggi_mm = h_px * MM_PER_PIXEL

                # 2. Volume (V) dalam ml (Prediksi YOLO)
                vol_yolo = h_px * ML_PER_PIXEL

                # 3. Error (e) dalam %
                # Rumus: |(Prediksi - Asli) / Asli| * 100
                if VOLUME_ASLI_ML > 0:
                    error_persen = abs((vol_yolo - VOLUME_ASLI_ML) / VOLUME_ASLI_ML) * 100
                else:
                    error_persen = 0.0

                # --- TAMPILAN TEKS (T, V, e) ---
                
                # Format string
                txt_T = f"T: {tinggi_mm:.2f} mm"
                txt_V = f"V: {vol_yolo:.2f} ml"
                txt_e = f"e: {error_persen:.2f} %"

                # Posisi Text
                base_x = int(x1) + 10
                base_y = int(y1) + 30
                gap = 30 # Jarak antar baris

                # Fungsi helper visual
                def draw_label(img, text, x, y, color):
                    # Outline Hitam (Tebal 4)
                    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4) 
                    # Isi Warna (Tebal 2)
                    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)   

                # 1. Tampilkan T (Tinggi) - Warna Putih
                draw_label(annotated_frame, txt_T, base_x, base_y, (255, 255, 255))
                
                # 2. Tampilkan V (Volume) - Warna Kuning/Cyan
                draw_label(annotated_frame, txt_V, base_x, base_y + gap, (0, 255, 255))

                # 3. Tampilkan e (Error) - Warna Hijau jika < 5%, Merah jika > 5%
                warnanya = (0, 255, 0) if error_persen < 10.0 else (0, 0, 255)
                draw_label(annotated_frame, txt_e, base_x, base_y + (gap*2), warnanya)

        out.write(annotated_frame)
        
        # Resize Preview
        preview = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
        cv2.imshow("OpenVINO Inference", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Selesai!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Point this to the OpenVINO folder or .xml file
    parser.add_argument("--weights", type=str, required=True, help="Path model (folder OpenVINO or .xml)")
    parser.add_argument("--source", type=str, required=True, help="Source video/webcam")
    parser.add_argument("--output", type=str, default="result_final.mp4", help="Output filename")
    
    # Default device CPU
    parser.add_argument("--device", type=str, default="CPU", help="Device: 'CPU', 'GPU' (Intel)")
    
    args = parser.parse_args()
    detect_video(args.weights, args.source, args.output, device=args.device)