import cv2
import argparse
from ultralytics import YOLO

def detect_video(weights, source, output, conf=0.25, device="CPU"):
    """
    Run object detection on an MP4 file using a custom YOLO model (OpenVINO supported).
    """
    # --- CALIBRATION CONFIG ---
    REF_PIXEL_HEIGHT = 900.0
    REF_REAL_HEIGHT = 4.5
    
    # FIXED: Was dividing height by itself. Corrected to real_height / pixel_height
    MM_PER_PIXEL = REF_REAL_HEIGHT / REF_PIXEL_HEIGHT 

    print(f"Loading model: {weights}")
    print(f"Device target: {device}")
    print(f"Calibration: 1 px = {MM_PER_PIXEL:.5f} mm")
    
    # Load YOLO model
    # Ultralytics automatically detects if 'weights' is a .pt file or an OpenVINO directory/XML
    try:
        model = YOLO(weights, task="detect")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Open video file
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
    print(f"💾 Output: {output}")
    print(f"⚙️ Confidence threshold: {conf}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        # device='CPU' or 'GPU' (Intel) works for OpenVINO
        # device=0 or 'cuda' will fail for OpenVINO models
        results = model.predict(frame, conf=conf, device=device, verbose=False)
        
        # Draw boxes on frame
        annotated_frame = results[0].plot()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                _, _, _, h = box.xywh[0].tolist()
                
                # Calculate pixel height and text info
                h_pixel = int(h)
                # Optional: Calculate real height based on your calibration
                h_mm = h_pixel * MM_PER_PIXEL
                
                text_info = f"H: {h_pixel}px"
                # text_info = f"H: {h_mm:.2f}mm" # Uncomment to use real units
                
                text_x = int(x1) + 5 
                text_y = int(y1) + 40

                # Draw Shadow (Thickness 4)
                cv2.putText(annotated_frame, text_info, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                                
                # Draw Text (Thickness 2)
                cv2.putText(annotated_frame, text_info, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(annotated_frame)

        # Smaller preview window (50% size)
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
    # Updated help text to indicate OpenVINO support
    parser.add_argument("--weights", type=str, required=True, help="Path to best_openvino_model/ directory or .xml file")
    parser.add_argument("--source", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--output", type=str, default="result.mp4", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    # Updated default device to CPU (Standard for OpenVINO)
    parser.add_argument("--device", type=str, default="CPU", help="'CPU', 'GPU' (Intel), or 'AUTO'")
    
    args = parser.parse_args()

    detect_video(args.weights, args.source, args.output, args.conf, args.device)