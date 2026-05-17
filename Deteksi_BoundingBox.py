import cv2
import argparse
from ultralytics import YOLO

def detect_video(weights, source, output, conf=0.25, device="CPU"):
    """
    Menjalankan deteksi objek murni dengan kotak dan class saja.
    """
    print(f"Loading model: {weights}")
    
    try:
        model = YOLO(weights, task="detect")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open video: {source}")
        return

    # Video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height_video))
    
    print(f"🎥 Processing: {source}")
    print(f"⚙️ Confidence threshold: {conf}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, conf=conf, device=device, verbose=False)
        
        # .plot() secara default menggambar Bounding Box dan Class Name
        annotated_frame = results[0].plot()
        
        # Simpan ke file
        out.write(annotated_frame)

        # Preview window (50% size)
        preview = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5)
        cv2.imshow("OpenVINO Detection", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Detection complete! Output saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path ke model")
    parser.add_argument("--source", type=str, required=True, help="Path ke video")
    parser.add_argument("--output", type=str, default="result.mp4", help="Output file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="CPU", help="'CPU', 'GPU', atau 'AUTO'")
    
    args = parser.parse_args()
    detect_video(args.weights, args.source, args.output, args.conf, args.device)