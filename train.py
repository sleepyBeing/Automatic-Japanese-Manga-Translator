from ultralytics import YOLO
import torch
import os

def main():
    # Device selection logic
    if torch.cuda.is_available():
        device = 0  
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:                                                        
        device = "cpu"
    
    print(f"Using device: {device}")

    # Load a model
    model = YOLO("yolov8n.pt") 

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data.yaml")

    # Training
    results = model.train(
        data="data.yaml",  # path to dataset YAML
        epochs=100,  # epoch
        imgsz=416,  # image size
        device=device,  
        name="manga_translator",  # experiment name
        plots=True,  # save plots
        batch=4, # batch size
        workers=1, # dataloader workers
        optimizer='Adam', 
        amp=False, 
        cache='ram', # cache images in RAM
        exist_ok=True, 
        max_det=300, # limit max detections per image
    )

    print("Training completed.")

if __name__ == "__main__":
    main()