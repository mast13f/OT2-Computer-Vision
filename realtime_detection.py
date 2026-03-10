"""
Real-time YOLO-NAS detection for Tips and Liquid.
Uses the trained model from Trained Models_NAS/ckpt_best.pth

Usage:
    # Live display from video file
    python realtime_detection.py --source /path/to/video.mp4

    # Save output video instead of displaying
    python realtime_detection.py --source /path/to/video.mp4 --save output.mp4

    # Webcam
    python realtime_detection.py --source 0

Controls (live display mode):
    q - Quit
    s - Save current frame
"""

import argparse
import time
import cv2
import torch
from super_gradients.training import models

# Detection classes (must match training order)
CLASSES = ["Tip", "Liquid"]

# Colors for each class (BGR)
COLORS = {"Tip": (0, 255, 0), "Liquid": (255, 0, 0)}


def load_model(checkpoint_path):
    """Load the trained YOLO-NAS model on CPU."""
    print(f"[1/3] Loading model from {checkpoint_path} ...")
    model = models.get(
        "yolo_nas_l",
        num_classes=len(CLASSES),
        checkpoint_path=checkpoint_path,
    )
    model = model.to("cpu")
    model.eval()
    print("[1/3] Model loaded successfully on CPU.")
    return model


def draw_detections(frame, pred, conf_threshold):
    """Draw bounding boxes and labels on the frame. Returns annotated frame and counts."""
    bboxes = pred.prediction.bboxes_xyxy
    confidences = pred.prediction.confidence
    labels = pred.prediction.labels

    tip_count = 0
    liquid_count = 0

    for bbox, conf, label in zip(bboxes, confidences, labels):
        if conf < conf_threshold:
            continue

        class_name = CLASSES[int(label)]
        color = COLORS[class_name]

        if class_name == "Tip":
            tip_count += 1
        else:
            liquid_count += 1

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{class_name} {conf:.2f}"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - ts[1] - 6), (x1 + ts[0], y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, tip_count, liquid_count


def run(source, checkpoint_path, conf, iou, save_path):
    model = load_model(checkpoint_path)

    # Open video source
    print(f"[2/3] Opening video source: {source}")
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source '{source}'")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[2/3] Video opened: {fw}x{fh}, {total_frames} frames, {src_fps:.1f} fps")

    # Set up video writer if saving
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, src_fps, (fw, fh))
        print(f"[3/3] Saving output to: {save_path}")
    else:
        print(f"[3/3] Showing live window. Press 'q' to quit, 's' to save a frame.")

    fps_list = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if not source.isdigit() and not save_path:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_num += 1
        t0 = time.time()

        # Run inference - model.predict on a single numpy array returns ImageDetectionPrediction
        preds = model.predict(frame, iou=iou, conf=conf, fuse_model=False)

        # preds is ImageDetectionPrediction with .prediction (DetectionPrediction)
        annotated, tip_count, liquid_count = draw_detections(frame, preds, conf)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])

        # Overlay info
        info = f"FPS: {avg_fps:.1f} | Tips: {tip_count} | Liquid: {liquid_count}"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(annotated)
            if frame_num % 50 == 0 or frame_num == 1:
                pct = frame_num / total_frames * 100 if total_frames > 0 else 0
                print(f"  Frame {frame_num}/{total_frames} ({pct:.0f}%) | FPS: {avg_fps:.1f} | Tips: {tip_count} Liquid: {liquid_count}")
        else:
            cv2.imshow("YOLO-NAS Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                sp = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(sp, annotated)
                print(f"Saved: {sp}")

    cap.release()
    if writer:
        writer.release()
        print(f"\nDone! Output saved to: {save_path}")
    else:
        cv2.destroyAllWindows()

    if fps_list:
        print(f"Average FPS: {sum(fps_list) / len(fps_list):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Real-time YOLO-NAS detection")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--model", type=str, default="Trained Models_NAS/ckpt_best.pth", help="Path to checkpoint")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--save", type=str, default=None, help="Save output to video file instead of displaying")
    args = parser.parse_args()

    print(f"Device: CPU (super_gradients works best on CPU/CUDA)")
    run(args.source, args.model, args.conf, args.iou, args.save)


if __name__ == "__main__":
    main()
