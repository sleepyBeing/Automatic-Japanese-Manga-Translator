
from ultralytics import YOLO
import torch

def main():
    # CUDA
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model

    # Train the model
    results = model.train(
        data="data.yaml",  # path to dataset YAML
        epochs=50,  # epoch
        imgsz=320,  # image size
        device=device,  #train on mps, cuda, or cpu
        name="manga_translator",  # experiment name
        plots=True,  # save plots
        batch=16, # batch size
        workers=0, # dataloader workers
        amp=False, # disable Automatic Mixed Precision (can be slow on MPS)
        cache='ram', # cache images in RAM for faster training
        exist_ok=True, # overwrite existing experiment
        max_det=50, # limit max detections per image (manga usually <50 boxes)
        conf=0.1, # increase confidence threshold to ignore noise
    )

    print("Training completed.")

if __name__ == "__main__":
    main()
