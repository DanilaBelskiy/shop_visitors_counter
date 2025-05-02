from ultralytics import YOLO
import cv2
import os


def process_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))

    output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        results = model.predict(
            source=frame,
            classes=[0],   # person
            conf=0.5,
            verbose=False
        )

        # Bounding boxes
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        # Counter refresh
        current = len(results[0].boxes)
        counter = max(counter, current)

    cap.release()
    writer.release()

    print(f"[DEBUG] Video saved to: {output_path}")
    print(f"[DEBUG] File exists: {os.path.exists(output_path)}")
    print(f"[DEBUG] File size: {os.path.getsize(output_path)} bytes")

    return counter


if __name__ == "__main__":

    # Input file
    input_path = "src/vid.mp4"

    # Output file
    output_path = "result/vid.mp4"

    # Load a model
    model = YOLO("yolo11n.pt")

    process_video(input_path, output_path, model)
