import os
import traceback
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, send_file, jsonify
from model import ModelYOLO
import mimetypes

from app_config import upload_folder, export_folder


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def handle_processing():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if not file or file.filename == '':
        return 'Invalid file submission', 400

    try:
        # Получаем имя файла и расширение
        filename = file.filename.lower()
        original_ext = filename.rsplit('.', 1)[-1]

        # Проверяем поддерживаемые форматы
        if original_ext not in ['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']:
            return 'Unsupported file format', 400

        # Генерируем уникальный ID для файла
        file_id = len(os.listdir(upload_folder))

        # 1. Сохраняем оригинальный файл
        original_filename = f"{file_id}.{original_ext}"
        original_path = os.path.join(upload_folder, original_filename)
        file.save(original_path)

        # 2. Обрабатываем файл в зависимости от типа
        if original_ext in ['mp4', 'avi', 'mov']:
            # Видеофайлы - конвертируем в MP4
            processed_ext = 'mp4'
            processed_filename = f"{file_id}.{processed_ext}"
            processed_path = os.path.join(export_folder, processed_filename)

            # Обрабатываем видео
            ship_count = model.process_video(original_path, processed_path)

            # Проверяем, что файл создан
            if not os.path.exists(processed_path):
                raise RuntimeError(f"Processed video file was not created at {processed_path}")

            file_type = 'video'

        # 4. Возвращаем результат
        return render_template('result.html',
                               original_filename=original_filename,
                               processed_filename=original_filename,
                               original_ext=original_ext,
                               processed_ext="mp4",
                               count=0)

    except Exception as e:
        # Подробное логирование ошибки
        app.logger.error(f"Error processing file: {str(e)}")
        app.logger.error(traceback.format_exc())

        # Удаляем временные файлы в случае ошибки
        if 'original_path' in locals() and os.path.exists(original_path):
            os.remove(original_path)
        if 'processed_path' in locals() and os.path.exists(processed_path):
            os.remove(processed_path)

        return render_template('error.html',
                               error_message=str(e),
                               error_details="File processing failed"), 500


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400

        file = request.files['frame']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.model(frame)
        detections = []

        if results[0].boxes:
            for box in results[0].boxes:
                if box.cls == model.classes[0]:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    detections.append({
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'confidence': box.conf.item() * 100
                    })

        return jsonify({'detections': detections})

    except Exception as e:
        app.logger.error(f"Frame processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500


mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/avi', '.avi')
mimetypes.add_type('video/quicktime', '.mov')


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(upload_folder, filename)


@app.route('/exports/<path:filename>')
def serve_export(filename):
    try:
        # Проверяем существование файла
        filepath = os.path.join(export_folder, filename)
        if not os.path.exists(filepath):
            return "File not found", 404

        # Устанавливаем заголовки для скачивания
        response = send_from_directory(
            export_folder,
            filename,
            as_attachment=True
        )

        # Для видео устанавливаем правильный MIME-тип
        if filename.lower().endswith('.mp4'):
            response.headers['Content-Type'] = 'video/mp4'

        return response

    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return "Internal server error", 500


@app.route('/report')
def generate_report():
    return send_file("stat.json", as_attachment=True, download_name='report.json')


@app.route('/download_result')
def send_result():
    return send_file(os.path.join(export_folder, '0.mp4'), as_attachment=True, download_name='result.mp4')


if __name__ == '__main__':

    model = ModelYOLO('yolo11n.pt')
    model.load_model()

    app.run(host='0.0.0.0', port=5000)
