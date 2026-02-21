import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from manga_ocr import MangaOcr
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm

def main():
    # 1. Initialize models
    print("Initializing models (this may take a moment if weights need to be downloaded)...")
    
    # YOLO for detection
    weights_path = "runs/detect/manga_translator/weights/best.pt"
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at {weights_path}. Run training or test.py first.")
        return
    model = YOLO(weights_path)
    
    mocr = MangaOcr()
    
    lama = SimpleLama()

    # 2. Setup source and output
    image_dir = "data/processed/test/images/" 
    output_dir = "runs/inference_results"
    os.makedirs(output_dir, exist_ok=True)

    test_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    if not test_images:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(test_images)} images. Starting full processing...")

    # 3. Process Loop
    for image_name in tqdm(test_images, desc="Processing Manga Pages"):
        target_image = os.path.join(image_dir, image_name)
        
        # Load images
        img_pil = Image.open(target_image).convert("RGB")
        w, h = img_pil.size
        
        # 4. Run Detection
        results = model.predict(target_image, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # 'text'
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    detections.append([int(c) for c in xyxy])

        # Sort detections: Right-to-Left (x2) and Top-to-Bottom (y1)
        detections.sort(key=lambda b: (-b[2], b[1]))

        # Create mask for LaMa 
        mask = np.zeros((h, w), dtype=np.uint8)
        ocr_results = []

        for x1, y1, x2, y2 in detections:
            # Crop and OCR
            crop = img_pil.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            text = mocr(crop)
            ocr_results.append({"box": [x1, y1, x2, y2], "text": text})
            
            # Update Mask 
            padding_mask = 5
            cv2.rectangle(mask, (max(0, x1-padding_mask), max(0, y1-padding_mask)), 
                          (min(w, x2+padding_mask), min(h, y2+padding_mask)), 255, -1)

        # 5. Refine Mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 6. Run Inpainting
        mask_pil = Image.fromarray(mask).convert("L")
        result_img_pil = lama(img_pil, mask_pil)
        
        # 7. Save Results
        save_path = os.path.join(output_dir, f"cleaned_{image_name}")
        result_img_pil.save(save_path)
        
        # Save OCR text
        with open(os.path.join(output_dir, f"ocr_{os.path.splitext(image_name)[0]}.txt"), "w", encoding="utf-8") as f:
            for res in ocr_results:
                f.write(f"Box {res['box']}: {res['text']}\n")

    print(f"\nProcessing Complete!")
    print(f"All results saved in: {output_dir}")

if __name__ == "__main__":
    main()
