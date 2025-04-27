# YOLO and Label Studio Active Learning System Setup Guide

This document provides detailed instructions on setting up and using a Docker-based active learning system integrating YOLO and Label Studio.

## Temp script
```bash
setup_kitti.sh 
```
it is a demo, which just use kitti-dataset to verify the workflow
There is a problem with environment variant in **project ID**.If you find out, please fix it.

## System Architecture

The system consists of two Docker containers:
1. **Label Studio Container**: For manual labeling and correction
2. **YOLO Service Container**: Runs YOLO model for prediction, training, and active learning

The containers communicate via Docker networking and exchange data through shared volumes.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU support): [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Project Structure

Create the following directory structure:

```
yolo-label-studio-project/
├── docker-compose.yml          # Docker Compose configuration
├── .env                        # Environment variables
├── shared-data/                # Shared data directory
│   ├── new_images/             # New images for labeling
│   ├── yolo_predictions/       # YOLO prediction results
│   ├── corrected_data/         # Corrected annotation data
│   └── yolo_training/          # Training results and models
├── label-studio-data/          # Label Studio data directory
└── yolo-service/               # YOLO service directory
    ├── Dockerfile              # YOLO service Dockerfile
    ├── requirements.txt        # Python dependencies
    └── active_learning.py      # Active learning code
```

## Setup Steps

### 1. Create Directory Structure

```bash
mkdir -p yolo-label-studio-project/shared-data/{new_images,yolo_predictions,corrected_data,yolo_training}
mkdir -p yolo-label-studio-project/label-studio-data
mkdir -p yolo-label-studio-project/yolo-service
```

### 2. Copy Configuration Files

Copy the provided code into:
- `docker-compose.yml`: Container configurations
- `.env`: Environment variables
- `yolo-service/Dockerfile`: YOLO service Dockerfile
- `yolo-service/requirements.txt`: Python dependencies
- `yolo-service/active_learning.py`: Active learning code

### 3. Configure Environment Variables

Edit `.env` to set:
- `LABEL_STUDIO_API_KEY`: Label Studio API key
- `PROJECT_ID`: Label Studio project ID (auto-created if not exists)
- `CUDA_VISIBLE_DEVICES`: GPU device ID (for GPU usage)

**How to Obtain Label Studio API Key:**
1. Run `docker-compose up -d label-studio`
2. Access Label Studio at `http://localhost:8080`
3. Login with default credentials:
   - Email: `admin@example.com`
   - Password: `admin`
4. Click user icon → "Account & Settings"
5. Under "Access Token" section, create/copy API key
6. Update `.env`:
   ```bash
   LABEL_STUDIO_API_KEY=your_copied_key
   ```
7. Restart containers:
   ```bash
   docker-compose restart
   ```

### 4. Start the System

```bash
cd yolo-label-studio-project
docker-compose up -d
```

Initial startup may take time for downloading and building containers.

### 5. Configure Label Studio

1. Access `http://localhost:8080`
2. Login with `admin@example.com` / `admin`
3. Create new project or use auto-created project
4. Obtain API key as described above
5. Update `.env` with `LABEL_STUDIO_API_KEY` and `PROJECT_ID`
6. Restart containers: `docker-compose restart`

## Workflow

### 1. Add Image Data

Place images in `shared-data/new_images/`:

```bash
cp /path/to/your/images/*.jpg yolo-label-studio-project/shared-data/new_images/
```

### 2. Automatic Processing

YOLO service will:
1. Detect new images
2. Generate predictions using YOLO
3. Upload pre-annotations to Label Studio
4. Monitor task completion

### 3. Label in Label Studio

1. Login to Label Studio
2. Navigate to project
3. Review YOLO pre-annotations
4. Correct annotations and submit

### 4. Automatic Model Updates

When all tasks are labeled, YOLO service will:
1. Export corrected data to `shared-data/corrected_data/`
2. Fine-tune YOLO model
3. Save updated model to `shared-data/yolo_training/active_learning_run/weights/`
4. Use updated model for next predictions

The system automatically cycles through:
- Model prediction → Human correction → Model update → Prediction...

## Inter-container Communication

1. **Network Communication**:
   - Dedicated `active-learning-network` created by Docker Compose
   - Label Studio hostname: `label-studio`
   - YOLO service accesses Label Studio API via `http://label-studio:8080`

2. **Data Sharing**:
   - `shared-data` mounted to both containers
   - YOLO predictions/training data stored in shared directory
   - File system-based data exchange

## Monitoring & Logs

View container logs:

```bash
# Label Studio logs
docker-compose logs -f label-studio

# YOLO service logs
docker-compose logs -f yolo-service
```

## Customization

### Custom Models & Labels

1. **Modify YOLO Model**:
   - Place custom model in `yolo-service`
   - Update `YOLO_MODEL_PATH` in `active_learning.py`

2. **Custom Labels**:
   - Configure labels in Label Studio UI
   - Modify `label_config` in `active_learning.py`

## Troubleshooting

### Common Issues

1. **Container Startup Failure**:
   - Verify Docker/Docker Compose installation
   - Check disk space and memory
   - Review logs: `docker-compose logs`

2. **Connection Issues**:
   - Ensure containers share the same network
   - Confirm Label Studio is running
   - Validate API key

3. **GPU Unavailable**:
   - Verify NVIDIA drivers and toolkit
   - Check GPU usage by other processes
   - Set `CUDA_VISIBLE_DEVICES=-1` in `.env` for CPU mode

4. **Data Import/Export Issues**:
   - Check shared directory permissions
   - Verify directory paths
   - Ensure sufficient disk space

## Extensions

Enhance the system with:

1. **Additional Annotation Formats**:
   - Modify import/export code

2. **Advanced Active Learning Strategies**:
   - Implement uncertainty sampling in `active_learning.py`

3. **Web Interface**:
   - Add monitoring/control UI container

4. **CI/CD Integration**:
   - Add automated testing/deployment

5. **Multi-model Support**:
   - Extend code to support multiple YOLO models

## Maintenance & Updates

Update components:

1. **Update Label Studio**:
   ```bash
   docker-compose pull label-studio
   docker-compose up -d
   ```

2. **Update YOLO Service**:
   ```bash
   docker-compose build yolo-service
   docker-compose up -d
   ```

## Security Considerations

1. Do not expose Label Studio publicly
2. Regularly update containers
3. Avoid storing sensitive credentials in `.env`
4. Set appropriate permissions for shared volumes