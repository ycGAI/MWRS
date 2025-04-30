import os
import json
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
import requests
import base64
from tqdm import tqdm
import glob
from pathlib import Path
import time

from ultralytics import YOLO

# Label Studio related
import label_studio_sdk
from label_studio_sdk import Client

# Get configuration from environment variables
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://label-studio:8080")
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "your_api_key_here")
# 确保环境变量正确处理
try:
    PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))
    print(f"Successfully initialized PROJECT_ID={PROJECT_ID}")
except Exception as e:
    print(f"Error getting PROJECT_ID from environment, using default: {e}")
    PROJECT_ID = 1

YOLO_MODEL_PATH = "yolov8n.pt"  # Path to pretrained YOLO model
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold
NEW_DATA_DIR = "/app/shared-data/new_images"  # New data directory
OUTPUT_DIR = "/app/shared-data/yolo_predictions"  # Directory to save YOLO prediction results
CORRECTED_DATA_DIR = "/app/shared-data/corrected_data"  # Directory for corrected data

# Ensure directories exist
os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CORRECTED_DATA_DIR, exist_ok=True)

class YOLOActiveLearning:
    def __init__(self):
        global PROJECT_ID        
        # Wait for Label Studio service to start
        self.wait_for_label_studio()
        
        self.model = YOLO(YOLO_MODEL_PATH)
        self.ls_client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
        
        # Ensure the project exists
        try:
            self.project = self.ls_client.get_project(PROJECT_ID)
            print(f"Successfully connected to project ID: {PROJECT_ID}")
        except Exception as e:
            print(f"Unable to get project {PROJECT_ID}, error: {e}")
            print("Trying to create a new project...")
            self.project = self.ls_client.create_project(
                title="YOLO Active Learning",
                description="Project using YOLO for active learning",
                label_config="""
                <View>
                  <Image name="image" value="$image"/>
                  <RectangleLabels name="label" toName="image">
                    <Label value="person" background="#FF0000"/>
                    <Label value="car" background="#00FF00"/>
                    <Label value="dog" background="#0000FF"/>
                    <!-- Add more labels as needed -->
                  </RectangleLabels>
                </View>
                """
            )
            PROJECT_ID = self.project.id
            print(f"Created a new project, ID: {PROJECT_ID}")
        
        # Load class mapping
        self.class_map = self.model.names
        
        # Create Label Studio class to YOLO class mapping
        self.create_class_mapping()
    
    def wait_for_label_studio(self, max_retries=30, retry_interval=10):
        """Wait for Label Studio service to be available"""
        print(f"Waiting for Label Studio service to start at {LABEL_STUDIO_URL}...")
        for i in range(max_retries):
            try:
                response = requests.get(f"{LABEL_STUDIO_URL}/health")
                if response.status_code == 200:
                    print("Label Studio service has started!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            
            print(f"Retry {i+1}/{max_retries}...")
            time.sleep(retry_interval)
        
        print("Unable to connect to Label Studio service, please check configuration.")
        return False
        
    def create_class_mapping(self):
        """Create Label Studio class to YOLO class mapping"""
        # Get labels from Label Studio
        ls_labels = self.project.params['label_config']
        # Parse XML to get labels (simplified version, might need adjustment based on actual label configuration)
        import re
        self.ls_labels = re.findall(r'value="([^"]+)"', ls_labels)
        
        # Create Label Studio label to YOLO class mapping
        self.ls_to_yolo = {}
        for idx, label in enumerate(self.ls_labels):
            if label in self.class_map.values():
                # Find corresponding class ID in YOLO
                for yolo_id, yolo_label in self.class_map.items():
                    if yolo_label == label:
                        self.ls_to_yolo[label] = yolo_id
            else:
                # If Label Studio has classes not in YOLO, we can extend YOLO classes
                # Here we assume Label Studio classes match YOLO classes
                self.ls_to_yolo[label] = idx
        
        # Create YOLO class to Label Studio label mapping
        self.yolo_to_ls = {v: k for k, v in self.ls_to_yolo.items()}
        
    def predict_batch(self, image_paths):
        """Use YOLO model to predict on a batch of images"""
        results = []
        
        for img_path in tqdm(image_paths, desc="YOLO prediction"):
            # Run model prediction
            result = self.model(img_path, conf=CONFIDENCE_THRESHOLD)
            
            # Save prediction results
            img_name = os.path.basename(img_path)
            results.append({
                "image_path": img_path,
                "image_name": img_name,
                "predictions": result
            })
            
        return results
    
    def convert_to_labelstudio_format(self, results):
        """Convert YOLO prediction results to Label Studio format"""
        tasks = []
        
        for result in results:
            img_path = result["image_path"]
            img_name = result["image_name"]
            prediction = result["predictions"][0]  # Get the first prediction result (single image)
            
            # Read image to get dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Prepare annotation data
            annotations = []
            
            # If there are predicted boxes
            if len(prediction.boxes) > 0:
                boxes = prediction.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
                cls = prediction.boxes.cls.cpu().numpy()  # Get classes
                conf = prediction.boxes.conf.cpu().numpy()  # Get confidence scores
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    class_id = int(cls[i])
                    confidence = float(conf[i])
                    
                    # Convert class ID to Label Studio label
                    if class_id in self.yolo_to_ls:
                        label = self.yolo_to_ls[class_id]
                    else:
                        label = f"class_{class_id}"
                    
                    # Label Studio uses relative coordinates (0-100%)
                    width = (x2 - x1) / img_width * 100
                    height = (y2 - y1) / img_height * 100
                    x = x1 / img_width * 100
                    y = y1 / img_height * 100
                    
                    # Create Label Studio format annotation
                    annotation = {
                        "id": f"result_{i}",
                        "type": "rectanglelabels",
                        "value": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "rotation": 0,
                            "rectanglelabels": [label]
                        },
                        "score": confidence,
                        "from_name": "label",
                        "to_name": "image",
                        "original_width": img_width,
                        "original_height": img_height
                    }
                    
                    annotations.append(annotation)
            
            # Create task
            with open(img_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            task = {
                "data": {
                    "image": f"data:image/jpeg;base64,{encoded_image}"
                },
                "predictions": [
                    {
                        "model_version": "YOLOv8",
                        "score": np.mean([a["score"] for a in annotations]) if annotations else 0,
                        "result": annotations
                    }
                ]
            }
            
            tasks.append(task)
        
        return tasks
    
    def upload_to_labelstudio(self, tasks):
        """Upload tasks to Label Studio"""
        for task in tqdm(tasks, desc="Uploading to Label Studio"):
            self.project.import_tasks([task])
            
        print(f"Successfully uploaded {len(tasks)} tasks to Label Studio")
    
    def export_corrected_data(self):
        """Export corrected data from Label Studio"""
        # Get completed tasks
        completed_tasks = self.project.get_tasks(filters={"status": "completed"})
        
        os.makedirs(f"{CORRECTED_DATA_DIR}/images", exist_ok=True)
        os.makedirs(f"{CORRECTED_DATA_DIR}/labels", exist_ok=True)
        
        for task in tqdm(completed_tasks, desc="Exporting corrected data"):
            # Get image data
            image_data = task["data"]["image"]
            if image_data.startswith("data:image/"):
                # Decode image from base64
                img_format = image_data.split(";")[0].split("/")[1]
                img_data = base64.b64decode(image_data.split(",")[1])
                img_name = f"image_{task['id']}.{img_format}"
                
                # Save image
                with open(f"{CORRECTED_DATA_DIR}/images/{img_name}", "wb") as f:
                    f.write(img_data)
                    
                # Process annotations
                if "annotations" in task:
                    annotation = task["annotations"][0]  # Get the latest annotation
                    result = annotation["result"]
                    
                    # Get image dimensions
                    img = Image.open(f"{CORRECTED_DATA_DIR}/images/{img_name}")
                    img_width, img_height = img.size
                    
                    # Create YOLO format annotation file
                    label_file = f"{CORRECTED_DATA_DIR}/labels/{os.path.splitext(img_name)[0]}.txt"
                    
                    with open(label_file, "w") as f:
                        for item in result:
                            if item["type"] == "rectanglelabels":
                                # Get label
                                label = item["value"]["rectanglelabels"][0]
                                
                                # Convert back to YOLO class ID
                                if label in self.ls_to_yolo:
                                    class_id = self.ls_to_yolo[label]
                                else:
                                    continue  # Skip unrecognized labels
                                
                                # Get bounding box coordinates (convert from percentages back to actual pixels)
                                x = item["value"]["x"] / 100 * img_width
                                y = item["value"]["y"] / 100 * img_height
                                width = item["value"]["width"] / 100 * img_width
                                height = item["value"]["height"] / 100 * img_height
                                
                                # Convert to YOLO format (center point coordinates and width/height, normalized)
                                x_center = (x + width / 2) / img_width
                                y_center = (y + height / 2) / img_height
                                width_norm = width / img_width
                                height_norm = height / img_height
                                
                                # Write YOLO format annotation
                                f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
        
        print(f"Successfully exported corrected data to {CORRECTED_DATA_DIR}")
        
    def create_dataset_yaml(self):
        """Create dataset YAML file needed for YOLO training"""
        dataset_yaml = {
            "path": os.path.abspath(CORRECTED_DATA_DIR),
            "train": "images",
            "val": "images",  # Simplified version, using the same images for validation
            "names": self.class_map
        }
        
        with open(f"{CORRECTED_DATA_DIR}/dataset.yaml", "w") as f:
            yaml.dump(dataset_yaml, f)
        
        print(f"Successfully created dataset configuration file: {CORRECTED_DATA_DIR}/dataset.yaml")
    
    def fine_tune_model(self, epochs=10, batch_size=16):
        """Fine-tune YOLO model using corrected data"""
        # Create dataset YAML
        self.create_dataset_yaml()
        
        # Fine-tune model
        print("Starting YOLO model fine-tuning...")
        self.model.train(
            data=f"{CORRECTED_DATA_DIR}/dataset.yaml",
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=0 if torch.cuda.is_available() else 'cpu',
            save=True,
            project='/app/shared-data/yolo_training',
            name='active_learning_run'
        )
        
        print("Model fine-tuning completed!")
        
        # Update current model
        self.model = YOLO('/app/shared-data/yolo_training/active_learning_run/weights/best.pt')
        
        return '/app/shared-data/yolo_training/active_learning_run/weights/best.pt'
    
    def monitor_label_studio(self, check_interval=60):
        """Monitor task completion status in Label Studio"""
        while True:
            # Get number of tasks
            tasks = self.project.get_tasks()
            total_tasks = len(tasks)
            
            if total_tasks == 0:
                print("No tasks to monitor.")
                time.sleep(check_interval)
                continue
            
            # Check completed tasks
            completed_tasks = self.project.get_tasks(filters={"status": "completed"})
            completed_count = len(completed_tasks)
            
            print(f"Label Studio task status: {completed_count}/{total_tasks} completed")
            
            # If all tasks are completed, start processing data
            if completed_count == total_tasks and total_tasks > 0:
                print("All tasks completed, starting data processing...")
                self.export_corrected_data()
                new_model_path = self.fine_tune_model(epochs=5)
                print(f"Active learning cycle completed! Updated model saved at: {new_model_path}")
                
                # Wait for new tasks
                print("Waiting for new tasks...")
            
            time.sleep(check_interval)
    
    def run_active_learning_cycle(self, image_paths=None):
        """Run a complete active learning cycle"""
        if not image_paths:
            # If no image paths provided, try to get them from the directory
            image_paths = glob.glob(f"{NEW_DATA_DIR}/*.jpg") + glob.glob(f"{NEW_DATA_DIR}/*.jpeg") + glob.glob(f"{NEW_DATA_DIR}/*.png")
        
        if not image_paths:
            print(f"No image files found in {NEW_DATA_DIR} directory.")
            print("Please place images to process in the shared directory, then run again.")
            return
        
        # 1. Use current model for prediction
        print("Step 1: Using YOLO model for prediction...")
        results = self.predict_batch(image_paths)
        
        # 2. Convert prediction results to Label Studio format
        print("Step 2: Converting prediction results to Label Studio format...")
        tasks = self.convert_to_labelstudio_format(results)
        
        # 3. Upload to Label Studio
        print("Step 3: Uploading tasks to Label Studio for manual correction...")
        self.upload_to_labelstudio(tasks)
        
        # 4. Monitor task completion in Label Studio
        print("Step 4: Starting to monitor Label Studio tasks...")
        self.monitor_label_studio()


# Main program
if __name__ == "__main__":
    print("Starting YOLO active learning system...")
    
    # Initialize
    active_learner = YOLOActiveLearning()
    
    # Periodically check for new images and process them
    while True:
        # Get images that need annotation
        image_paths = glob.glob(f"{NEW_DATA_DIR}/*.jpg") + glob.glob(f"{NEW_DATA_DIR}/*.jpeg") + glob.glob(f"{NEW_DATA_DIR}/*.png")
        
        if image_paths:
            print(f"Found {len(image_paths)} new image files, starting processing...")
            active_learner.run_active_learning_cycle(image_paths)
        else:
            print(f"No new image files found in {NEW_DATA_DIR} directory. Waiting...")
            time.sleep(60)  # Check once per minute