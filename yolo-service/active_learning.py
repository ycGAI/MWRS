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

# 假设您已经安装了YOLOv5
# pip install -U ultralytics
from ultralytics import YOLO

# Label Studio相关
import label_studio_sdk
from label_studio_sdk import Client

# 从环境变量获取配置
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://label-studio:8080")
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "your_api_key_here")
PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))

YOLO_MODEL_PATH = "yolov8n.pt"  # 预训练的YOLO模型路径
CONFIDENCE_THRESHOLD = 0.25  # 检测置信度阈值
NEW_DATA_DIR = "/app/shared-data/new_images"  # 新数据目录
OUTPUT_DIR = "/app/shared-data/yolo_predictions"  # YOLO预测结果保存目录
CORRECTED_DATA_DIR = "/app/shared-data/corrected_data"  # 修正后的数据目录

# 确保目录存在
os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CORRECTED_DATA_DIR, exist_ok=True)

class YOLOActiveLearning:
    def __init__(self):
        # 等待Label Studio服务启动
        self.wait_for_label_studio()
        
        self.model = YOLO(YOLO_MODEL_PATH)
        self.ls_client = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
        
        # 确保项目存在
        try:
            self.project = self.ls_client.get_project(PROJECT_ID)
            print(f"成功连接到项目ID: {PROJECT_ID}")
        except Exception as e:
            print(f"无法获取项目 {PROJECT_ID}, 错误: {e}")
            print("尝试创建新项目...")
            self.project = self.ls_client.create_project(
                title="YOLO Active Learning",
                description="使用YOLO进行主动学习的项目",
                label_config="""
                <View>
                  <Image name="image" value="$image"/>
                  <RectangleLabels name="label" toName="image">
                    <Label value="person" background="#FF0000"/>
                    <Label value="car" background="#00FF00"/>
                    <Label value="dog" background="#0000FF"/>
                    <!-- 根据需要添加更多标签 -->
                  </RectangleLabels>
                </View>
                """
            )
            PROJECT_ID = self.project.id
            print(f"创建了新项目，ID: {PROJECT_ID}")
        
        # 加载类别映射
        self.class_map = self.model.names
        
        # 创建Label Studio类别到YOLO类别的映射
        self.create_class_mapping()
    
    def wait_for_label_studio(self, max_retries=30, retry_interval=10):
        """等待Label Studio服务可用"""
        print(f"等待Label Studio服务在 {LABEL_STUDIO_URL} 启动...")
        for i in range(max_retries):
            try:
                response = requests.get(f"{LABEL_STUDIO_URL}/health")
                if response.status_code == 200:
                    print("Label Studio服务已启动!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            
            print(f"重试 {i+1}/{max_retries}...")
            time.sleep(retry_interval)
        
        print("无法连接到Label Studio服务，请检查配置。")
        return False
        
    def create_class_mapping(self):
        """创建Label Studio类别到YOLO类别的映射"""
        # 获取Label Studio中的标签
        ls_labels = self.project.params['label_config']
        # 解析XML获取标签（简化版，可能需要根据实际标签配置调整）
        import re
        self.ls_labels = re.findall(r'value="([^"]+)"', ls_labels)
        
        # 创建Label Studio标签到YOLO类别的映射
        self.ls_to_yolo = {}
        for idx, label in enumerate(self.ls_labels):
            if label in self.class_map.values():
                # 找到YOLO中对应的类别ID
                for yolo_id, yolo_label in self.class_map.items():
                    if yolo_label == label:
                        self.ls_to_yolo[label] = yolo_id
            else:
                # 如果Label Studio中有YOLO没有的类别，可以扩展YOLO类别
                # 这里假设Label Studio中的类别与YOLO类别一致
                self.ls_to_yolo[label] = idx
        
        # 创建YOLO类别到Label Studio标签的映射
        self.yolo_to_ls = {v: k for k, v in self.ls_to_yolo.items()}
        
    def predict_batch(self, image_paths):
        """使用YOLO模型对一批图像进行预测"""
        results = []
        
        for img_path in tqdm(image_paths, desc="YOLO预测"):
            # 运行模型预测
            result = self.model(img_path, conf=CONFIDENCE_THRESHOLD)
            
            # 保存预测结果
            img_name = os.path.basename(img_path)
            results.append({
                "image_path": img_path,
                "image_name": img_name,
                "predictions": result
            })
            
        return results
    
    def convert_to_labelstudio_format(self, results):
        """将YOLO预测结果转换为Label Studio格式"""
        tasks = []
        
        for result in results:
            img_path = result["image_path"]
            img_name = result["image_name"]
            prediction = result["predictions"][0]  # 获取第一个预测结果（单张图片）
            
            # 读取图像获取尺寸
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # 准备标注数据
            annotations = []
            
            # 如果有预测框
            if len(prediction.boxes) > 0:
                boxes = prediction.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
                cls = prediction.boxes.cls.cpu().numpy()  # 获取类别
                conf = prediction.boxes.conf.cpu().numpy()  # 获取置信度
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    class_id = int(cls[i])
                    confidence = float(conf[i])
                    
                    # 将类别ID转换为Label Studio标签
                    if class_id in self.yolo_to_ls:
                        label = self.yolo_to_ls[class_id]
                    else:
                        label = f"class_{class_id}"
                    
                    # Label Studio使用的是相对坐标（0-100%）
                    width = (x2 - x1) / img_width * 100
                    height = (y2 - y1) / img_height * 100
                    x = x1 / img_width * 100
                    y = y1 / img_height * 100
                    
                    # 创建Label Studio格式的标注
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
            
            # 创建任务
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
        """将任务上传到Label Studio"""
        for task in tqdm(tasks, desc="上传到Label Studio"):
            self.project.import_tasks([task])
            
        print(f"成功上传 {len(tasks)} 个任务到Label Studio")
    
    def export_corrected_data(self):
        """从Label Studio导出修正后的数据"""
        # 获取已完成的任务
        completed_tasks = self.project.get_tasks(filters={"status": "completed"})
        
        os.makedirs(f"{CORRECTED_DATA_DIR}/images", exist_ok=True)
        os.makedirs(f"{CORRECTED_DATA_DIR}/labels", exist_ok=True)
        
        for task in tqdm(completed_tasks, desc="导出修正后的数据"):
            # 获取图像数据
            image_data = task["data"]["image"]
            if image_data.startswith("data:image/"):
                # 从base64解码图像
                img_format = image_data.split(";")[0].split("/")[1]
                img_data = base64.b64decode(image_data.split(",")[1])
                img_name = f"image_{task['id']}.{img_format}"
                
                # 保存图像
                with open(f"{CORRECTED_DATA_DIR}/images/{img_name}", "wb") as f:
                    f.write(img_data)
                    
                # 处理标注
                if "annotations" in task:
                    annotation = task["annotations"][0]  # 获取最新的标注
                    result = annotation["result"]
                    
                    # 获取图像尺寸
                    img = Image.open(f"{CORRECTED_DATA_DIR}/images/{img_name}")
                    img_width, img_height = img.size
                    
                    # 创建YOLO格式的标注文件
                    label_file = f"{CORRECTED_DATA_DIR}/labels/{os.path.splitext(img_name)[0]}.txt"
                    
                    with open(label_file, "w") as f:
                        for item in result:
                            if item["type"] == "rectanglelabels":
                                # 获取标签
                                label = item["value"]["rectanglelabels"][0]
                                
                                # 转换回YOLO的类别ID
                                if label in self.ls_to_yolo:
                                    class_id = self.ls_to_yolo[label]
                                else:
                                    continue  # 跳过不认识的标签
                                
                                # 获取边界框坐标（从百分比转回实际像素）
                                x = item["value"]["x"] / 100 * img_width
                                y = item["value"]["y"] / 100 * img_height
                                width = item["value"]["width"] / 100 * img_width
                                height = item["value"]["height"] / 100 * img_height
                                
                                # 转换为YOLO格式（中心点坐标和宽高，归一化）
                                x_center = (x + width / 2) / img_width
                                y_center = (y + height / 2) / img_height
                                width_norm = width / img_width
                                height_norm = height / img_height
                                
                                # 写入YOLO格式标注
                                f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
        
        print(f"成功导出修正后的数据到 {CORRECTED_DATA_DIR}")
        
    def create_dataset_yaml(self):
        """创建YOLO训练所需的数据集YAML文件"""
        dataset_yaml = {
            "path": os.path.abspath(CORRECTED_DATA_DIR),
            "train": "images",
            "val": "images",  # 简化版，使用相同的图像进行验证
            "names": self.class_map
        }
        
        with open(f"{CORRECTED_DATA_DIR}/dataset.yaml", "w") as f:
            yaml.dump(dataset_yaml, f)
        
        print(f"成功创建数据集配置文件: {CORRECTED_DATA_DIR}/dataset.yaml")
    
    def fine_tune_model(self, epochs=10, batch_size=16):
        """使用修正后的数据微调YOLO模型"""
        # 创建数据集YAML
        self.create_dataset_yaml()
        
        # 微调模型
        print("开始微调YOLO模型...")
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
        
        print("模型微调完成！")
        
        # 更新当前模型
        self.model = YOLO('/app/shared-data/yolo_training/active_learning_run/weights/best.pt')
        
        return '/app/shared-data/yolo_training/active_learning_run/weights/best.pt'
    
    def monitor_label_studio(self, check_interval=60):
        """监控Label Studio中的任务完成情况"""
        while True:
            # 获取任务数量
            tasks = self.project.get_tasks()
            total_tasks = len(tasks)
            
            if total_tasks == 0:
                print("没有任务需要监控。")
                time.sleep(check_interval)
                continue
            
            # 检查已完成的任务
            completed_tasks = self.project.get_tasks(filters={"status": "completed"})
            completed_count = len(completed_tasks)
            
            print(f"Label Studio中的任务状态: {completed_count}/{total_tasks} 已完成")
            
            # 如果所有任务都已完成，开始处理数据
            if completed_count == total_tasks and total_tasks > 0:
                print("所有任务已完成，开始处理数据...")
                self.export_corrected_data()
                new_model_path = self.fine_tune_model(epochs=5)
                print(f"主动学习循环完成！更新后的模型保存在: {new_model_path}")
                
                # 等待新任务
                print("等待新任务...")
            
            time.sleep(check_interval)
    
    def run_active_learning_cycle(self, image_paths=None):
        """运行一个完整的主动学习循环"""
        if not image_paths:
            # 如果没有提供图像路径，尝试从目录中获取
            image_paths = glob.glob(f"{NEW_DATA_DIR}/*.jpg") + glob.glob(f"{NEW_DATA_DIR}/*.jpeg") + glob.glob(f"{NEW_DATA_DIR}/*.png")
        
        if not image_paths:
            print(f"在 {NEW_DATA_DIR} 目录中没有找到图像文件。")
            print("请将要处理的图像放入共享目录，然后再次运行。")
            return
        
        # 1. 使用当前模型进行预测
        print("第1步：使用YOLO模型进行预测...")
        results = self.predict_batch(image_paths)
        
        # 2. 将预测结果转换为Label Studio格式
        print("第2步：转换预测结果为Label Studio格式...")
        tasks = self.convert_to_labelstudio_format(results)
        
        # 3. 上传到Label Studio
        print("第3步：上传任务到Label Studio进行人工修正...")
        self.upload_to_labelstudio(tasks)
        
        # 4. 监控Label Studio中任务的完成情况
        print("第4步：开始监控Label Studio任务...")
        self.monitor_label_studio()


# 主程序
if __name__ == "__main__":
    print("启动YOLO主动学习系统...")
    
    # 初始化
    active_learner = YOLOActiveLearning()
    
    # 定期检查新图像并处理
    while True:
        # 获取需要标注的图像
        image_paths = glob.glob(f"{NEW_DATA_DIR}/*.jpg") + glob.glob(f"{NEW_DATA_DIR}/*.jpeg") + glob.glob(f"{NEW_DATA_DIR}/*.png")
        
        if image_paths:
            print(f"发现 {len(image_paths)} 个新图像文件，开始处理...")
            active_learner.run_active_learning_cycle(image_paths)
        else:
            print(f"在 {NEW_DATA_DIR} 目录中没有找到新图像文件。等待中...")
            time.sleep(60)  # 每分钟检查一次