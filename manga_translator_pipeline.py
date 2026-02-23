import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from manga_ocr import MangaOcr
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from translator_engine import MangaTranslator
from renderer import MangaRenderer

def main():
    # 1. Initialize models
    print("Initializing models...")
    
    # YOLO for detection
    weights_path = "runs/detect/manga_translator/weights/best.pt"
    if not os.path.exists(weights_path):
        # Fallback to a common location or error out
        weights_path = "yolov8n.pt" 
        print(f"Warning: Could not find weights at runs/detect/manga_translator/weights/best.pt. Using default yolov8n.pt.")
    
    detector = YOLO(weights_path)
    ocr = MangaOcr()
    lama = SimpleLama()
    translator = MangaTranslator()
    renderer = MangaRenderer()

    # 2. Setup source and output
    image_dir = "data/processed/test/images/" 
    output_dir = "runs/translation_results"
    os.makedirs(output_dir, exist_ok=True)

    test_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    if not test_images:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(test_images)} images. Starting pipeline...")

    # 3. Process Loop
    for image_name in tqdm(test_images, desc="Processing Manga Pages"):
        target_path = os.path.join(image_dir, image_name)
        
        # Load images
        img_pil = Image.open(target_path).convert("RGB")
        w, h = img_pil.size
        
        # 4. Run Detection
        results = detector.predict(target_path, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # 'text'
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    detections.append([int(c) for c in xyxy])

        # Sort detections: Right-to-Left and Top-to-Bottom
        detections.sort(key=lambda b: (-b[2], b[1]))

        # Create mask for LaMa 
        mask = np.zeros((h, w), dtype=np.uint8)
        processed_regions = []

        for x1, y1, x2, y2 in detections:
            # Crop and OCR
            crop = img_pil.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            ja_text = ocr(crop)
            
            # Translate
            en_text = translator.translate(ja_text)
            
            processed_regions.append({
                "box": [x1, y1, x2, y2], 
                "ja_text": ja_text,
                "en_text": en_text
            })
            
            # Update Mask 
            padding_mask = 5
            cv2.rectangle(mask, (max(0, x1-padding_mask), max(0, y1-padding_mask)), 
                          (min(w, x2+padding_mask), min(h, y2+padding_mask)), 255, -1)

        # 5. Inpainting (Remove Japanese text)
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_pil = Image.fromarray(mask_dilated).convert("L")
        
        inpainted_img = lama(img_pil, mask_pil)
        
        # 6. Render
        final_img = inpainted_img.copy()
        for region in processed_regions:
            box = region["box"]
            en_text = region["en_text"]
            if en_text.strip():
                final_img = renderer.render_text(final_img, en_text, box)

        # 7. Save Results
        save_path = os.path.join(output_dir, f"translated_{image_name}")
        final_img.save(save_path)
        
    print(f"\nProcessing Complete!")
    print(f"All results saved in: {output_dir}")

if __name__ == "__main__":
    main()
