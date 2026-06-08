from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8n.pt")


def detect_image(image_path, save_path=None, people_only=False):
    """Run detection on an image file, annotate it, and optionally save.

    Returns: (counts_dict, total_count, annotated_image_array)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(image)

    counts = defaultdict(int)
    annotated = image.copy()
    names = {}
    # Try to obtain class names from model or results
    try:
        names = model.names
    except Exception:
        try:
            names = results[0].names
        except Exception:
            names = {}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if people_only and cls != 0:
                continue
            name = names.get(cls, str(cls))
            counts[name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated,
                        f"{counts[name]} {name}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

    total = sum(counts.values())
    cv2.putText(annotated,
                f"Total: {total}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        try:
            annotated_resized = cv2.resize(annotated, (900, 600))
        except Exception:
            annotated_resized = annotated
        cv2.imwrite(save_path, annotated_resized)

    return dict(counts), total, annotated


if __name__ == "__main__":
    # Demo usage (keeps original behavior but via function)
    try:
        counts, total, annotated = detect_image("deepan.jpeg")
        annotated_resized = cv2.resize(annotated, (900, 600))
        cv2.imshow("People Detection", annotated_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)