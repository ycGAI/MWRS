#!/bin/bash

# Set color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== YOLO + Label Studio + KITTI Dataset Startup Script ===${NC}"

# Check for necessary software
echo -e "${YELLOW}Checking for necessary software...${NC}"
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is missing, please install Docker first${NC}" >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}Docker Compose is missing, please install it first${NC}" >&2; exit 1; }

# Create project directory structure
echo -e "${YELLOW}Creating project directory structure...${NC}"
mkdir -p shared-data/{new_images,yolo_predictions,corrected_data,yolo_training,kitti}
mkdir -p label-studio-data

# Set directory permissions
echo -e "${YELLOW}Setting directory permissions...${NC}"
chmod -R 755 shared-data 2>/dev/null || echo -e "${YELLOW}Some directory permissions cannot be modified, but this does not affect functionality${NC}"
chmod -R 755 label-studio-data 2>/dev/null || echo -e "${YELLOW}Some directory permissions cannot be modified, but this does not affect functionality${NC}"
chmod -R 755 yolo-service 2>/dev/null || echo -e "${YELLOW}Some directory permissions cannot be modified, but this does not affect functionality${NC}"

# Fix PROJECT_ID issue in modified_active_learning.py
echo -e "${YELLOW}Fixing environment variable handling issue...${NC}"
if grep -q "PROJECT_ID = int(os.environ.get(\"PROJECT_ID\"" yolo-service/modified_active_learning.py; then
    # Create backup
    cp yolo-service/modified_active_learning.py yolo-service/modified_active_learning.py.bak
    
    # Replace environment variable handling code
    sed -i 's/PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))/# Enhanced environment variable handling\ntry:\n    PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))\n    print(f"Successfully initialized PROJECT_ID={PROJECT_ID}")\nexcept Exception as e:\n    print(f"Error getting PROJECT_ID from environment, using default value 1")\n    PROJECT_ID = 1/' yolo-service/modified_active_learning.py
    
    echo -e "${GREEN}Fixed environment variable handling in modified_active_learning.py${NC}"
else
    echo -e "${YELLOW}modified_active_learning.py is already updated, no fix needed${NC}"
fi

# Check for KITTI dataset
echo -e "${YELLOW}Checking for KITTI dataset...${NC}"
KITTI_IMAGES_DIR="shared-data/kitti/training/image_2"
KITTI_LABELS_DIR="shared-data/kitti/training/label_2"

# Count existing image and label files
IMAGE_COUNT=0
LABEL_COUNT=0
if [ -d "$KITTI_IMAGES_DIR" ]; then
    IMAGE_COUNT=$(find "$KITTI_IMAGES_DIR" -name "*.png" | wc -l)
fi
if [ -d "$KITTI_LABELS_DIR" ]; then
    LABEL_COUNT=$(find "$KITTI_LABELS_DIR" -name "*.txt" | wc -l)
fi

if [ $IMAGE_COUNT -gt 0 ] && [ $LABEL_COUNT -gt 0 ]; then
    echo -e "${GREEN}Found existing KITTI dataset: ${IMAGE_COUNT} image files, ${LABEL_COUNT} label files${NC}"
    read -p "Do you want to use the existing KITTI data? [Y/n]: " use_existing
    
    if [[ $use_existing == "n" || $use_existing == "N" ]]; then
        echo -e "${YELLOW}Will re-download the KITTI dataset...${NC}"
        DOWNLOAD_KITTI=true
    else
        echo -e "${GREEN}Using existing KITTI dataset${NC}"
        DOWNLOAD_KITTI=false
    fi
else
    echo -e "${YELLOW}Complete KITTI dataset not found${NC}"
    read -p "Do you want to download a sample of the KITTI dataset? (Approximately 1GB of data will be downloaded) [y/N]: " download_input
    
    if [[ $download_input == "y" || $download_input == "Y" ]]; then
        DOWNLOAD_KITTI=true
    else
        DOWNLOAD_KITTI=false
    fi
fi

# Download KITTI dataset if needed
if [ "$DOWNLOAD_KITTI" = true ]; then
    echo -e "${YELLOW}Downloading KITTI dataset sample...${NC}"
    # Create temporary directory and enter
    mkdir -p temp_kitti
    cd temp_kitti
    
    # Download sample images (actual KITTI dataset includes all images, here only downloading a sample)
    echo -e "${YELLOW}Downloading image data (large files, please be patient)...${NC}"
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
    
    # Download labels
    echo -e "${YELLOW}Downloading label data...${NC}"
    wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
    
    # Extract files
    echo -e "${YELLOW}Extracting data...${NC}"
    unzip -q data_object_image_2.zip
    unzip -q data_object_label_2.zip
    
    # Create KITTI required directory structure
    mkdir -p ../shared-data/kitti/training/image_2
    mkdir -p ../shared-data/kitti/training/label_2
    
    # Copy files (to save space, only copy the first 100 files)
    echo -e "${YELLOW}Copying data to working directory...${NC}"
    find training/image_2/ -name "*.png" | head -100 | xargs -I {} cp {} ../shared-data/kitti/training/image_2/
    find training/label_2/ -name "*.txt" | head -100 | xargs -I {} cp {} ../shared-data/kitti/training/label_2/
    
    # Clean up
    cd ..
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -rf temp_kitti
    
    echo -e "${GREEN}KITTI dataset sample prepared${NC}"
else
    if [ $IMAGE_COUNT -eq 0 ] || [ $LABEL_COUNT -eq 0 ]; then
        echo -e "${YELLOW}Skipping KITTI dataset download...${NC}"
        echo -e "${YELLOW}Please manually place the KITTI dataset in the shared-data/kitti directory with the following structure:${NC}"
        echo -e "${YELLOW}   shared-data/kitti/training/image_2/    - contains images${NC}"
        echo -e "${YELLOW}   shared-data/kitti/training/label_2/    - contains labels${NC}"
    fi
fi

# Copy KITTI data to new_images directory (ensure YOLO service can find them)
echo -e "${YELLOW}Preparing KITTI data for processing...${NC}"
# Clear new_images directory to avoid processing too many images
echo -e "${YELLOW}Clearing processing directory for a new batch...${NC}"
rm -f shared-data/new_images/*.png shared-data/new_images/*.jpg shared-data/new_images/*.jpeg 2>/dev/null

if [ -d "$KITTI_IMAGES_DIR" ] && [ "$(find $KITTI_IMAGES_DIR -name '*.png' | wc -l)" -gt 0 ]; then
    echo -e "${YELLOW}Copying KITTI images to processing directory...${NC}"
    # Only copy the first 20 images to avoid processing too many at once
    find "$KITTI_IMAGES_DIR" -name "*.png" | head -20 | xargs -I {} cp {} shared-data/new_images/
    echo -e "${GREEN}Prepared $(find shared-data/new_images -name '*.png' | wc -l) images for processing${NC}"
else
    echo -e "${YELLOW}KITTI image data not found, skipping copy${NC}"
fi

# Check for YOLOv8 pre-trained weights
echo -e "${YELLOW}Checking for YOLOv8 pre-trained weights...${NC}"
if [ ! -f "yolo-service/yolov8n.pt" ]; then
    echo -e "${YELLOW}Downloading YOLOv8 pre-trained weights...${NC}"
    wget -c https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolo-service/yolov8n.pt
    chmod 644 yolo-service/yolov8n.pt 2>/dev/null || echo -e "${YELLOW}Unable to set permissions, but this does not affect functionality${NC}"
else
    echo -e "${YELLOW}YOLOv8 pre-trained weights already exist, skipping download${NC}"
fi

# Ensure .env file exists and is correctly configured
echo -e "${YELLOW}Checking environment variable configuration...${NC}"
if [ -f ".env" ]; then
    # Check if API key is set to default
    if grep -q "LABEL_STUDIO_API_KEY=your_api_key_here" .env; then
        echo -e "${YELLOW}Default API key found, please update after Label Studio starts${NC}"
    fi
else
    # Create new .env file
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOL
LABEL_STUDIO_API_KEY=your_api_key_here
PROJECT_ID=1
CUDA_VISIBLE_DEVICES=0
EOL
    echo -e "${GREEN}.env file created, please update the API key after Label Studio starts${NC}"
fi

# Ask whether to restart containers
read -p "Do you want to restart Docker containers? [y/N]: " restart_containers
if [[ $restart_containers == "y" || $restart_containers == "Y" ]]; then
    echo -e "${YELLOW}Stopping and restarting Docker containers...${NC}"
    docker-compose down
    docker-compose up -d
else
    # Check if containers are running
    LABEL_STUDIO_RUNNING=$(docker-compose ps | grep label-studio | grep "Up" | wc -l)
    YOLO_SERVICE_RUNNING=$(docker-compose ps | grep yolo-service | grep "Up" | wc -l)
    
    if [ $LABEL_STUDIO_RUNNING -eq 0 ] || [ $YOLO_SERVICE_RUNNING -eq 0 ]; then
        echo -e "${YELLOW}Containers not running, starting them...${NC}"
        docker-compose up -d
    else
        echo -e "${GREEN}Containers are already running, no need to restart${NC}"
    fi
fi

# Wait for Label Studio to start
echo -e "${YELLOW}Waiting for Label Studio to start...${NC}"
max_retries=30
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:8080/health | grep -q "ok"; then
        echo -e "${GREEN}Label Studio started successfully!${NC}"
        break
    fi
    echo -e "${YELLOW}Waiting for Label Studio to start (${retry_count}/${max_retries})...${NC}"
    sleep 10
    retry_count=$((retry_count+1))
done

if [ $retry_count -eq $max_retries ]; then
    echo -e "${RED}Label Studio startup timed out, please check the logs: docker-compose logs label-studio${NC}"
    docker-compose logs label-studio | tail -50
    exit 1
fi

# Display instructions to get API key
echo -e "${GREEN}=== System Started ===${NC}"
echo -e "${YELLOW}Access Label Studio to set up the API key:${NC}"
echo -e "  1. Open your browser and go to http://localhost:8080"
echo -e "  2. Log in with default credentials: username admin@example.com, password admin"
echo -e "  3. If no project is visible, create one named 'KITTI Object Detection' for object detection"
echo -e "     - Click 'Create Project'"
echo -e "     - Enter name: KITTI Object Detection"
echo -e "     - Select type: Object Detection with Bounding Boxes"
echo -e "     - Add labels: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc"
echo -e "  4. Note the project ID number in the URL (e.g., '1' in http://localhost:8080/projects/1/data)"
echo -e "  5. Click the user icon in the top right -> 'Account & Settings'"
echo -e "  6. Create an API key in the 'Access Token' section"
echo -e "  7. Copy and update the .env file:"
echo -e "     LABEL_STUDIO_API_KEY=<your API key>"
echo -e "     PROJECT_ID=<your project ID>"
echo -e "  8. Restart the containers: docker-compose restart"
echo -e "${GREEN}=== Enjoy the Active Learning System with YOLO and Label Studio! ===${NC}"
echo -e "${YELLOW}View system logs:${NC}"
echo -e "  - Label Studio logs: docker-compose logs -f label-studio"
echo -e "  - YOLO service logs: docker-compose logs -f yolo-service"
echo -e ""
echo -e "${YELLOW}System Usage Process:${NC}"
echo -e "  1. KITTI data has been placed in the shared-data/kitti directory"
echo -e "  2. The YOLO service will automatically detect and process images in new_images"
echo -e "  3. Complete annotation corrections in Label Studio"
echo -e "  4. The system will automatically update the model for the next batch of predictions"
echo -e ""
echo -e "${YELLOW}Tip: You can run the following command to check the YOLO processing progress:${NC}"
echo -e "  docker-compose logs -f yolo-service"