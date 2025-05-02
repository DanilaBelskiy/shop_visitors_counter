from ultralytics import YOLO
import cv2
import os

from datetime import datetime


class ModelYOLO:
    def __init__(self, model_name):

        self.model_name = model_name
        self.model = None
        self.classes = [0]   # Person class

    def load_model(self):
        self.model = YOLO(self.model_name)

    def process_frame(self, image):
        return self.model.predict(source=image, classes=self.classes, conf=0.5, verbose=False)

    def process_video(self, input_path, output_path):

        start_time = datetime.now()

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
            results = self.process_frame(frame)

            # Bounding boxes
            annotated_frame = results[0].plot()
            writer.write(annotated_frame)

            # Counter refresh
            current = len(results[0].boxes)
            counter = max(counter, current)

        cap.release()
        writer.release()

        print(f"[DEBUG] Video processed")
        print(f"[DEBUG] Processing time = {datetime.now() - start_time}")
        print(f"[DEBUG] Result saved to: {output_path}")

        return counter
