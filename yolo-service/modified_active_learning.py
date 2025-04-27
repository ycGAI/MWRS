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
import shutil

from ultralytics import YOLO

# Label Studio related
import label_studio_sdk
from label_studio_sdk import Client

# Get configuration from environment variables
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://label-studio:8080")
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "your_api_key_here")
# 环境变量处理增强
try:
    PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))
    print(f"Successfully initialized PROJECT_ID={PROJECT_ID}")
except Exception as e:
    print(f"Error getting PROJECT_ID from environment, using default value 1")
    PROJECT_ID = 1

# Path to pretrained YOLO model - can use YOLOv8 pretrained model
YOLO_MODEL_PATH = "yolov8n.pt"  
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold
NEW_DATA_DIR = "/app/shared-data/new_images"  # New data directory
OUTPUT_DIR = "/app/shared-data/yolo_predictions"  # Directory to save YOLO prediction results
CORRECTED_DATA_DIR = "/app/shared-data/corrected_data"  # Directory for corrected data
KITTI_DATA_DIR = "/app/shared-data/kitti_data"  # Directory for KITTI data

# Ensure directories exist
os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CORRECTED_DATA_DIR, exist_ok=True)
os.makedirs(KITTI_DATA_DIR, exist_ok=True)
os.makedirs(f"{KITTI_DATA_DIR}/images", exist_ok=True)
os.makedirs(f"{KITTI_DATA_DIR}/labels", exist_ok=True)

# KITTI class names - update based on the specific KITTI dataset you're using
KITTI_CLASSES = {
    0: "Car",
    1: "Van",
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram",
    7: "Misc",
    8: "DontCare"
}

# Map KITTI classes to COCO classes for YOLOv8 pretrained model
KITTI_TO_COCO = {
    "Car": 2,         # car maps to car in COCO
    "Van": 2,         # van maps to car in COCO
    "Truck": 7,       # truck maps to truck in COCO
    "Pedestrian": 0,  # pedestrian maps to person in COCO
    "Person_sitting": 0,  # person_sitting maps to person in COCO
    "Cyclist": 1,     # cyclist maps to bicycle in COCO
    "Tram": 6,        # tram maps to train in COCO
    "Misc": 0,        # misc gets default mapping
    "DontCare": 0     # DontCare gets default mapping
}

class YOLOActiveLearning:
    def __init__(self):
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
                title="KITTI Object Detection with YOLO",
                description="Project using YOLO for active learning on KITTI dataset",
                label_config="""
                <View>
                  <Image name="image" value="$image"/>
                  <RectangleLabels name="label" toName="image">
                    <Label value="Car" background="#FF0000"/>
                    <Label value="Van" background="#00FF00"/>
                    <Label value="Truck" background="#0000FF"/>
                    <Label value="Pedestrian" background="#FFFF00"/>
                    <Label value="Person_sitting" background="#FF00FF"/>
                    <Label value="Cyclist" background="#00FFFF"/>
                    <Label value="Tram" background="#FFA500"/>
                    <Label value="Misc" background="#808080"/>
                    <Label value="DontCare" background="#A0A0A0"/>
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
        # Parse XML to get labels (simplified version)
        import re
        self.ls_labels = re.findall(r'value="([^"]+)"', ls_labels)
        
        # Create Label Studio label to YOLO class mapping
        self.ls_to_yolo = {}
        for label in self.ls_labels:
            if label in KITTI_CLASSES.values():
                # Map KITTI class to COCO class ID in YOLO
                if label in KITTI_TO_COCO:
                    self.ls_to_yolo[label] = KITTI_TO_COCO[label]
                else:
                    # Default mapping if not found
                    self.ls_to_yolo[label] = 0
        
        # Create YOLO class to Label Studio label mapping
        self.yolo_to_ls = {}
        for kitti_class, yolo_id in KITTI_TO_COCO.items():
            if yolo_id in self.class_map:
                self.yolo_to_ls[yolo_id] = kitti_class
            
        print("Class mappings created:")
        print(f"Label Studio to YOLO: {self.ls_to_yolo}")
        print(f"YOLO to Label Studio: {self.yolo_to_ls}")
    
    def process_kitti_dataset(self, kitti_dir):
        """Process KITTI dataset and prepare it for use"""
        images_dir = os.path.join(kitti_dir, "image_2")
        labels_dir = os.path.join(kitti_dir, "label_2")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"KITTI directory structure not found at {kitti_dir}")
            print("Expected: {kitti_dir}/image_2 and {kitti_dir}/label_2")
            return False
        
        # Process each image and label file
        image_files = glob.glob(f"{images_dir}/*.png")
        
        for img_path in tqdm(image_files, desc="Processing KITTI dataset"):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_path):
                # Copy image to new_images directory for processing
                shutil.copy(img_path, os.path.join(NEW_DATA_DIR, img_name))
                
                # Also save a copy in our KITTI directory
                shutil.copy(img_path, os.path.join(KITTI_DATA_DIR, "images", img_name))
                
                # Convert KITTI label to YOLO format
                self.convert_kitti_to_yolo(label_path, img_path, os.path.join(KITTI_DATA_DIR, "labels", f"{base_name}.txt"))
        
        print(f"Processed {len(image_files)} KITTI images and labels")
        return True
    
    def convert_kitti_to_yolo(self, kitti_label_path, img_path, output_path):
        """Convert KITTI label format to YOLO format"""
        # Read image to get dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        with open(kitti_label_path, 'r') as f:
            kitti_labels = f.readlines()
        
        with open(output_path, 'w') as outfile:
            for line in kitti_labels:
                parts = line.strip().split()
                if len(parts) < 15:  # KITTI format has at least 15 parts
                    continue
                
                kitti_class = parts[0]
                
                # Skip DontCare or classes we don't want to track
                if kitti_class == 'DontCare':
                    continue
                    
                # Map KITTI class to YOLO class
                if kitti_class in KITTI_TO_COCO:
                    yolo_class = KITTI_TO_COCO[kitti_class]
                else:
                    # Skip classes we don't have a mapping for
                    continue
                
                # KITTI format: [left, top, right, bottom] in pixels
                left = float(parts[4])
                top = float(parts[5])
                right = float(parts[6])
                bottom = float(parts[7])
                
                # Convert to YOLO format: [x_center, y_center, width, height] normalized
                x_center = ((left + right) / 2) / img_width
                y_center = ((top + bottom) / 2) / img_height
                width = (right - left) / img_width
                height = (bottom - top) / img_height
                
                # Write YOLO format
                outfile.write(f"{yolo_class} {x_center} {y_center} {width} {height}\n")
    
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
                    
                    # Map YOLO class to KITTI class for Label Studio
                    if class_id in self.yolo_to_ls:
                        label = self.yolo_to_ls[class_id]
                    else:
                        # Skip classes we can't map
                        continue
                    
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
            "names": {v: k for k, v in KITTI_TO_COCO.items()}  # Map back from COCO IDs to KITTI classes
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
            name='kitti_active_learning_run'
        )
        
        print("Model fine-tuning completed!")
        
        # Update current model
        self.model = YOLO('/app/shared-data/yolo_training/kitti_active_learning_run/weights/best.pt')
        
        return '/app/shared-data/yolo_training/kitti_active_learning_run/weights/best.pt'
    
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
    print("Starting YOLO active learning system for KITTI dataset...")
    
    # Initialize
    active_learner = YOLOActiveLearning()
    
    # Check if KITTI dataset has been provided in the shared directory
    kitti_dir = "/app/shared-data/kitti"
    if os.path.exists(kitti_dir):
        print(f"Found KITTI dataset directory at {kitti_dir}, processing...")
        active_learner.process_kitti_dataset(kitti_dir)
    
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