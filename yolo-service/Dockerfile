FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制需求文件并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/new_images /app/yolo_predictions /app/corrected_data /app/yolo_training

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 下载预训练的YOLOv8模型
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 运行脚本
CMD ["python", "active_learning.py"]