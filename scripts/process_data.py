import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
IMAGES_DIR = RAW_DATA_DIR / "images"
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"

# YOLO classes
CLASSES = {"text": 0, "face": 1}

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_book(book_name):
    xml_file = ANNOTATIONS_DIR / f"{book_name}.xml"
    if not xml_file.exists():
        print(f"Warning: Annotation file for {book_name} not found.")
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    processed_files = []

    for page in root.findall(".//page"):
        page_index = page.get("index")
        width = int(page.get("width"))
        height = int(page.get("height"))
        
        # Image file name extraction
        # Manga109 structure: images/BookName/001.jpg
        image_name = f"{int(page_index):03d}.jpg" 
        src_image_path = IMAGES_DIR / book_name / image_name
        
        if not src_image_path.exists():
             # Try png if jpg doesn't exist, though usually jpg
            image_name = f"{int(page_index):03d}.png"
            src_image_path = IMAGES_DIR / book_name / image_name
            if not src_image_path.exists():
                continue

        label_data = []
        
        # Extract Text
        for text in page.findall("text"):
            xmin = int(text.get("xmin"))
            ymin = int(text.get("ymin"))
            xmax = int(text.get("xmax"))
            ymax = int(text.get("ymax"))
            
            b = (xmin, xmax, ymin, ymax)
            bb = convert_box((width, height), b)
            label_data.append(f"{CLASSES['text']} {' '.join(f'{a:.6f}' for a in bb)}")

        # Extract Faces
        for face in page.findall("face"):
            xmin = int(face.get("xmin"))
            ymin = int(face.get("ymin"))
            xmax = int(face.get("xmax"))
            ymax = int(face.get("ymax"))
            
            b = (xmin, xmax, ymin, ymax)
            bb = convert_box((width, height), b)
            label_data.append(f"{CLASSES['face']} {' '.join(f'{a:.6f}' for a in bb)}")
            
        if label_data:
            processed_files.append((src_image_path, label_data))
            
    return processed_files

def main():
    # Setup directories
    for split in ["train", "val", "test"]:
        (PROCESSED_DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Get list of books
    if not IMAGES_DIR.exists():
        print(f"Error: {IMAGES_DIR} does not exist. Please place your 'images' folder in 'data/raw/'.")
        return

    books = [d.name for d in IMAGES_DIR.iterdir() if d.is_dir()]
    random.shuffle(books)
    
    # Split books (not pages) to avoid data leakage
    # 80% train, 10% val, 10% test
    train_split = int(len(books) * 0.8)
    val_split = int(len(books) * 0.9)
    
    train_books = books[:train_split]
    val_books = books[train_split:val_split]
    test_books = books[val_split:]
    
    splits = [("train", train_books), ("val", val_books), ("test", test_books)]
    
    print(f"Found {len(books)} books.")
    print(f"Training on {len(train_books)} books.")
    print(f"Validating on {len(val_books)} books.")
    print(f"Testing on {len(test_books)} books.")

    for split_name, split_books in splits:
        print(f"Processing {split_name} set...")
        for book in tqdm(split_books):
            files = process_book(book)
            for src_image, labels in files:
                # Copy image
                # Construct unique filename: BookName_PageNum.jpg
                dst_image_name = f"{book}_{src_image.name}"
                dst_image_path = PROCESSED_DATA_DIR / split_name / "images" / dst_image_name
                shutil.copy(src_image, dst_image_path)
                
                # Write labels
                dst_label_path = PROCESSED_DATA_DIR / split_name / "labels" / f"{dst_image_name.rsplit('.', 1)[0]}.txt"
                with open(dst_label_path, "w") as f:
                    f.write("\n".join(labels))

    print("Data processing complete!")

if __name__ == "__main__":
    main()
