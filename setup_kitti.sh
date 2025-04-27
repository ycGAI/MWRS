#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== YOLO + Label Studio + KITTI 数据集启动脚本 ===${NC}"

# 检查是否安装了必要的软件
echo -e "${YELLOW}检查必要软件...${NC}"
command -v docker >/dev/null 2>&1 || { echo -e "${RED}缺少Docker，请先安装Docker${NC}" >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}缺少Docker Compose，请先安装${NC}" >&2; exit 1; }

# 创建必要的目录结构
echo -e "${YELLOW}创建项目目录结构...${NC}"
mkdir -p shared-data/{new_images,yolo_predictions,corrected_data,yolo_training,kitti}
mkdir -p label-studio-data

# 确保目录权限正确
echo -e "${YELLOW}设置目录权限...${NC}"
chmod -R 755 shared-data 2>/dev/null || echo -e "${YELLOW}部分目录权限无法修改，但这不影响功能${NC}"
chmod -R 755 label-studio-data 2>/dev/null || echo -e "${YELLOW}部分目录权限无法修改，但这不影响功能${NC}"
chmod -R 755 yolo-service 2>/dev/null || echo -e "${YELLOW}部分目录权限无法修改，但这不影响功能${NC}"

# 修复modified_active_learning.py文件中的PROJECT_ID问题
echo -e "${YELLOW}修复环境变量处理问题...${NC}"
if grep -q "PROJECT_ID = int(os.environ.get(\"PROJECT_ID\"" yolo-service/modified_active_learning.py; then
    # 创建备份
    cp yolo-service/modified_active_learning.py yolo-service/modified_active_learning.py.bak
    
    # 替换环境变量处理代码
    sed -i 's/PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))/# 环境变量处理增强\ntry:\n    PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))\n    print(f"Successfully initialized PROJECT_ID={PROJECT_ID}")\nexcept Exception as e:\n    print(f"Error getting PROJECT_ID from environment, using default value 1")\n    PROJECT_ID = 1/' yolo-service/modified_active_learning.py
    
    echo -e "${GREEN}已修复modified_active_learning.py中的环境变量处理${NC}"
else
    echo -e "${YELLOW}modified_active_learning.py已经更新，无需修复${NC}"
fi

# 检查KITTI数据集是否已经存在
echo -e "${YELLOW}检查KITTI数据集...${NC}"
KITTI_IMAGES_DIR="shared-data/kitti/training/image_2"
KITTI_LABELS_DIR="shared-data/kitti/training/label_2"

# 计算现有图像和标签文件数
IMAGE_COUNT=0
LABEL_COUNT=0
if [ -d "$KITTI_IMAGES_DIR" ]; then
    IMAGE_COUNT=$(find "$KITTI_IMAGES_DIR" -name "*.png" | wc -l)
fi
if [ -d "$KITTI_LABELS_DIR" ]; then
    LABEL_COUNT=$(find "$KITTI_LABELS_DIR" -name "*.txt" | wc -l)
fi

if [ $IMAGE_COUNT -gt 0 ] && [ $LABEL_COUNT -gt 0 ]; then
    echo -e "${GREEN}发现已有KITTI数据集: ${IMAGE_COUNT}个图像文件, ${LABEL_COUNT}个标签文件${NC}"
    read -p "是否使用现有KITTI数据？[Y/n]: " use_existing
    
    if [[ $use_existing == "n" || $use_existing == "N" ]]; then
        echo -e "${YELLOW}将重新下载KITTI数据集...${NC}"
        DOWNLOAD_KITTI=true
    else
        echo -e "${GREEN}使用现有KITTI数据集${NC}"
        DOWNLOAD_KITTI=false
    fi
else
    echo -e "${YELLOW}未发现完整的KITTI数据集${NC}"
    read -p "是否下载KITTI数据集示例？（大约需要下载1GB数据）[y/N]: " download_input
    
    if [[ $download_input == "y" || $download_input == "Y" ]]; then
        DOWNLOAD_KITTI=true
    else
        DOWNLOAD_KITTI=false
    fi
fi

# 下载KITTI数据集（如果需要）
if [ "$DOWNLOAD_KITTI" = true ]; then
    echo -e "${YELLOW}下载KITTI数据集示例...${NC}"
    # 创建临时目录并进入
    mkdir -p temp_kitti
    cd temp_kitti
    
    # 下载示例图像（实际KITTI数据集包含所有图像，这里仅下载部分示例）
    echo -e "${YELLOW}下载图像数据（较大文件，请耐心等待）...${NC}"
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
    
    # 下载标签
    echo -e "${YELLOW}下载标签数据...${NC}"
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
    
    # 解压文件
    echo -e "${YELLOW}解压数据...${NC}"
    unzip -q data_object_image_2.zip
    unzip -q data_object_label_2.zip
    
    # 创建KITTI所需的目录结构
    mkdir -p ../shared-data/kitti/training/image_2
    mkdir -p ../shared-data/kitti/training/label_2
    
    # 复制文件（为了节省空间，仅复制前100个文件）
    echo -e "${YELLOW}复制数据到工作目录...${NC}"
    find training/image_2/ -name "*.png" | head -100 | xargs -I {} cp {} ../shared-data/kitti/training/image_2/
    find training/label_2/ -name "*.txt" | head -100 | xargs -I {} cp {} ../shared-data/kitti/training/label_2/
    
    # 清理
    cd ..
    echo -e "${YELLOW}清理临时文件...${NC}"
    rm -rf temp_kitti
    
    echo -e "${GREEN}KITTI数据集示例准备完成${NC}"
else
    if [ $IMAGE_COUNT -eq 0 ] || [ $LABEL_COUNT -eq 0 ]; then
        echo -e "${YELLOW}跳过KITTI数据集下载...${NC}"
        echo -e "${YELLOW}请手动将KITTI数据集放在 shared-data/kitti 目录下，确保包含以下结构：${NC}"
        echo -e "${YELLOW}   shared-data/kitti/training/image_2/    - 包含图像${NC}"
        echo -e "${YELLOW}   shared-data/kitti/training/label_2/    - 包含标签${NC}"
    fi
fi

# 将KITTI数据拷贝到新图像目录(确保YOLO服务能发现它们)
echo -e "${YELLOW}准备KITTI数据用于处理...${NC}"
# 清空new_images目录，避免处理过多图像
echo -e "${YELLOW}清空处理目录，准备新的处理批次...${NC}"
rm -f shared-data/new_images/*.png shared-data/new_images/*.jpg shared-data/new_images/*.jpeg 2>/dev/null

if [ -d "$KITTI_IMAGES_DIR" ] && [ "$(find $KITTI_IMAGES_DIR -name '*.png' | wc -l)" -gt 0 ]; then
    echo -e "${YELLOW}拷贝KITTI图像到处理目录...${NC}"
    # 仅拷贝前20个图像进行处理，避免一次处理太多
    find "$KITTI_IMAGES_DIR" -name "*.png" | head -20 | xargs -I {} cp {} shared-data/new_images/
    echo -e "${GREEN}已准备$(find shared-data/new_images -name '*.png' | wc -l)个图像用于处理${NC}"
else
    echo -e "${YELLOW}未找到KITTI图像数据，跳过拷贝${NC}"
fi

# 检查YOLOv8预训练权重
echo -e "${YELLOW}检查YOLOv8预训练权重...${NC}"
if [ ! -f "yolo-service/yolov8n.pt" ]; then
    echo -e "${YELLOW}下载YOLOv8预训练权重...${NC}"
    wget -c https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolo-service/yolov8n.pt
    chmod 644 yolo-service/yolov8n.pt 2>/dev/null || echo -e "${YELLOW}无法设置权限，但不影响功能${NC}"
else
    echo -e "${YELLOW}YOLOv8预训练权重已存在，跳过下载${NC}"
fi

# 确保.env文件存在并正确配置
echo -e "${YELLOW}检查环境变量配置...${NC}"
if [ -f ".env" ]; then
    # 检查API密钥是否存在
    if grep -q "LABEL_STUDIO_API_KEY=your_api_key_here" .env; then
        echo -e "${YELLOW}发现默认API密钥，请在Label Studio启动后更新${NC}"
    fi
else
    # 创建新的.env文件
    echo -e "${YELLOW}创建.env文件...${NC}"
    cat > .env << EOL
LABEL_STUDIO_API_KEY=your_api_key_here
PROJECT_ID=1
CUDA_VISIBLE_DEVICES=0
EOL
    echo -e "${GREEN}已创建.env文件，请在Label Studio启动后更新API密钥${NC}"
fi

# 询问是否需要重启容器
read -p "是否重新启动Docker容器？[y/N]: " restart_containers
if [[ $restart_containers == "y" || $restart_containers == "Y" ]]; then
    echo -e "${YELLOW}停止并重启Docker容器...${NC}"
    docker-compose down
    docker-compose up -d
else
    # 检查容器是否在运行
    LABEL_STUDIO_RUNNING=$(docker-compose ps | grep label-studio | grep "Up" | wc -l)
    YOLO_SERVICE_RUNNING=$(docker-compose ps | grep yolo-service | grep "Up" | wc -l)
    
    if [ $LABEL_STUDIO_RUNNING -eq 0 ] || [ $YOLO_SERVICE_RUNNING -eq 0 ]; then
        echo -e "${YELLOW}检测到容器未运行，正在启动...${NC}"
        docker-compose up -d
    else
        echo -e "${GREEN}容器已在运行，无需重启${NC}"
    fi
fi

# 等待Label Studio启动
echo -e "${YELLOW}正在等待Label Studio启动...${NC}"
max_retries=30
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:8080/health | grep -q "ok"; then
        echo -e "${GREEN}Label Studio已启动成功！${NC}"
        break
    fi
    echo -e "${YELLOW}等待Label Studio启动 (${retry_count}/${max_retries})...${NC}"
    sleep 10
    retry_count=$((retry_count+1))
done

if [ $retry_count -eq $max_retries ]; then
    echo -e "${RED}Label Studio启动超时，请检查日志：docker-compose logs label-studio${NC}"
    docker-compose logs label-studio | tail -50
    exit 1
fi

# 显示获取API密钥的说明
echo -e "${GREEN}=== 系统已启动 ===${NC}"
echo -e "${YELLOW}访问Label Studio设置API密钥：${NC}"
echo -e "  1. 打开浏览器访问 http://localhost:8080"
echo -e "  2. 使用默认凭证登录：用户名 admin@example.com，密码 admin"
echo -e "  3. 如果没有看到项目，创建一个名为'KITTI Object Detection'的对象检测项目"
echo -e "     - 点击'Create Project'"
echo -e "     - 输入名称：KITTI Object Detection"
echo -e "     - 选择类型：Object Detection with Bounding Boxes"
echo -e "     - 添加标签：Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc"
echo -e "  4. 记下URL中的项目ID数字 (例如：http://localhost:8080/projects/1/data 中的'1')"
echo -e "  5. 点击右上角用户图标 -> 'Account & Settings'"
echo -e "  6. 在'Access Token'部分创建API密钥"
echo -e "  7. 复制并更新.env文件："
echo -e "     LABEL_STUDIO_API_KEY=<你的API密钥>"
echo -e "     PROJECT_ID=<你看到的项目ID>"
echo -e "  8. 重启容器：docker-compose restart"
echo -e "${GREEN}=== 享受YOLO和Label Studio的主动学习系统! ===${NC}"
echo -e "${YELLOW}系统日志查看：${NC}"
echo -e "  - Label Studio日志: docker-compose logs -f label-studio"
echo -e "  - YOLO服务日志: docker-compose logs -f yolo-service"
echo -e ""
echo -e "${YELLOW}系统使用流程：${NC}"
echo -e "  1. KITTI数据已放入shared-data/kitti目录"
echo -e "  2. YOLO服务会自动检测并处理new_images中的图像"
echo -e "  3. 在Label Studio中完成标注校正工作"
echo -e "  4. 系统会自动更新模型并用于下一批预测"
echo -e ""
echo -e "${YELLOW}提示：可以运行以下命令查看YOLO处理进度：${NC}"
echo -e "  docker-compose logs -f yolo-service"