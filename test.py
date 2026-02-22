from ultralytics import YOLO
import os

def test():
    # Load the best weights from your recent training
    weights_path = "runs/detect/manga_translator/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at {weights_path}")
        print("Make sure your training has finished or produced a best.pt file.")
        return

    model = YOLO(weights_path)
    
    # Run evaluation on the test set defined in data.yaml
    print("Starting evaluation on test set...")
    results = model.val(data="data.yaml", split='test', name="manga_test_results")
    
    print("\n" + "="*30)
    print("Testing completed.")
    print(f"Results saved in: runs/detect/manga_test_results")
    print("="*30)

if __name__ == "__main__":
    test()
