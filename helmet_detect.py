import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml # To read class names from data.yaml

# --- CONFIGURATION ---
# Path to your trained model weights
model_path = r'D:\Helmet_detection\weights\best.pt'

# Path to your data.yaml to get class names
data_yaml_path = r'D:\Helmet_detection\data.yaml' # Or 'corrected_data.yaml' if that's what you used

# Path to a sample image for testing
sample_image_path = r'D:\Helmet_detection\test_images' # Make sure this image exists!
# --- END CONFIGURATION ---

def load_class_names(yaml_path):
    """Loads class names from the data.yaml file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def run_inference_and_display(image_path, model_path, class_names):
    """Loads model, runs inference, and displays results."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found at {image_path}")
        return

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path) # Load your custom trained model

    print(f"Performing inference on: {image_path}")
    results = model(image_path) # Perform inference

    for r in results:
        im_bgr = r.plot() # plot() returns an image with bounding boxes drawn
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for matplotlib

        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title('YOLOv8 Detection Result')
        plt.show()

        # Print detection details
        print("Detection details:")
        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = class_names[class_id] # Use loaded class names
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  Class: {class_name}, Confidence: {confidence:.2f}, Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        else:
            print("  No detections found.")

        # Save the result image (optional)
        output_dir = 'output_detections'
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        result_image_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(result_image_path, im_bgr)
        print(f"\nDetection result image saved to: {result_image_path}")

if __name__ == '__main__':
    class_names = load_class_names(data_yaml_path)
    run_inference_and_display(sample_image_path, model_path, class_names)