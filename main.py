from model import ModelYOLO


if __name__ == "__main__":

    filename = "smol_vid.mp4"

    # Input file
    input_path = f"src/{filename}"

    # Output file
    output_path = f"result/{filename}"

    # Load a model
    model = ModelYOLO('yolo11n.pt')
    model.load_model()

    # Process video
    model.process_video(input_path, output_path)
