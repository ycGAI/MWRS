# YOLO与Label Studio主动学习系统设置指南

本文档提供了如何设置和使用基于Docker的YOLO与Label Studio主动学习系统的详细说明。

## 系统架构

该系统由两个Docker容器组成：
1. **Label Studio容器**：用于人工标注和修正
2. **YOLO服务容器**：运行YOLO模型用于预测、训练和主动学习

两个容器通过Docker网络进行通信，并通过共享卷来交换数据。

## 前提条件

- Docker和Docker Compose已安装
- NVIDIA容器工具包（如果使用GPU）：[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 项目结构

请按照以下目录结构创建项目文件：

```
yolo-label-studio-project/
├── docker-compose.yml          # Docker Compose配置文件
├── .env                        # 环境变量配置
├── shared-data/                # 共享数据目录
│   ├── new_images/             # 存放需要标注的新图像
│   ├── yolo_predictions/       # 存放YOLO预测结果
│   ├── corrected_data/         # 存放修正后的数据
│   └── yolo_training/          # 存放训练结果和模型
├── label-studio-data/          # Label Studio数据目录
└── yolo-service/               # YOLO服务目录
    ├── Dockerfile              # YOLO服务的Dockerfile
    ├── requirements.txt        # Python依赖列表
    └── active_learning.py      # 主动学习代码
```

## 设置步骤

### 1. 创建目录结构

```bash
mkdir -p yolo-label-studio-project/shared-data/{new_images,yolo_predictions,corrected_data,yolo_training}
mkdir -p yolo-label-studio-project/label-studio-data
mkdir -p yolo-label-studio-project/yolo-service
```

### 2. 复制配置文件

将提供的代码复制到以下文件中：
- `docker-compose.yml`：容器配置
- `.env`：环境变量
- `yolo-service/Dockerfile`：YOLO服务的Dockerfile
- `yolo-service/requirements.txt`：Python依赖
- `yolo-service/active_learning.py`：主动学习代码

### 3. 配置环境变量

编辑`.env`文件，设置以下变量：
- `LABEL_STUDIO_API_KEY`：Label Studio API密钥
- `PROJECT_ID`：Label Studio项目ID（如果不存在会自动创建）
- `CUDA_VISIBLE_DEVICES`：GPU设备ID（如果使用GPU）
如何获得Label Studio API:
运行 docker-compose up -d label-studio
在浏览器中访问Label Studio界面：http://localhost:8080
使用默认凭据登录：

用户名：admin@example.com
密码：admin


登录后，点击右上角的用户图标（通常显示您的用户名或头像）
在下拉菜单中选择"Account & Settings"（账户与设置）
在账户设置页面，查找"Access Token"或"API Keys"部分
您可能会看到一个现有的API密钥，或者需要点击"Create New Token"/"Create API Key"按钮来生成一个新的密钥
生成密钥后，将其复制并保存到您的项目的.env文件中，更新LABEL_STUDIO_API_KEY变量：
LABEL_STUDIO_API_KEY=您复制的密钥

保存.env文件后，需要重启容器以使更改生效：
bashdocker-compose restart

### 4. 启动系统

```bash
cd yolo-label-studio-project
docker-compose up -d
```

首次启动时，容器会下载并构建，这可能需要一些时间。

### 5. 设置Label Studio

1. 打开浏览器访问`http://localhost:8080`
2. 使用默认凭据登录：`admin@example.com` / `admin`
3. 创建一个新项目或使用YOLO服务自动创建的项目
4. 获取API密钥：
   - 点击右上角的用户图标
   - 选择"Account & Settings"
   - 在"Access Token"部分获取或创建API密钥
5. 更新`.env`文件中的`LABEL_STUDIO_API_KEY`和`PROJECT_ID`
6. 重启容器：`docker-compose restart`

## 使用流程

### 1. 添加图像数据

将需要标注的图像文件放入`shared-data/new_images/`目录：

```bash
cp /path/to/your/images/*.jpg yolo-label-studio-project/shared-data/new_images/
```

### 2. 自动处理

YOLO服务容器会：
1. 自动检测新图像
2. 使用YOLO模型进行预测
3. 将预测结果上传到Label Studio
4. 监控Label Studio中的任务完成情况

### 3. 在Label Studio中标注

1. 登录Label Studio
2. 进入项目
3. 查看带有YOLO预标注的任务
4. 修正预标注并提交

### 4. 自动更新模型

一旦所有任务都被标注完成，YOLO服务会：
1. 导出修正后的标注数据到`shared-data/corrected_data/`目录
2. 使用修正后的数据微调YOLO模型
3. 保存更新后的模型到`shared-data/yolo_training/active_learning_run/weights/`目录
4. 将更新后的模型用于下一轮预测

系统会自动循环这个过程，实现主动学习：
- 模型预测 → 人工修正 → 模型更新 → 模型预测...

## 容器间通信原理

两个容器的通信基于以下机制：

1. **网络通信**：
   - Docker Compose创建了名为`active-learning-network`的网络
   - Label Studio容器的主机名被设置为`label-studio`
   - YOLO服务通过`http://label-studio:8080`访问Label Studio API

2. **数据共享**：
   - `shared-data`目录被挂载到两个容器中
   - YOLO服务将预测结果和训练数据保存到共享目录
   - 两个容器通过文件系统共享数据

## 监控和日志

查看容器日志：

```bash
# 查看Label Studio日志
docker-compose logs -f label-studio

# 查看YOLO服务日志
docker-compose logs -f yolo-service
```

## 自定义模型和标签

要自定义YOLO模型和标签：

1. **修改YOLO模型**：
   - 将预训练的YOLO模型文件放在`yolo-service`目录中
   - 修改`active_learning.py`中的`YOLO_MODEL_PATH`变量

2. **自定义Label Studio标签**：
   - 在Label Studio界面中创建项目时自定义标签配置
   - 或修改`active_learning.py`中的`label_config`参数

## 疑难解答

### 常见问题

1. **容器无法启动**：
   - 检查Docker和Docker Compose是否正确安装
   - 检查是否有足够的磁盘空间和内存
   - 查看日志确定具体错误：`docker-compose logs`

2. **YOLO服务无法连接到Label Studio**：
   - 确保两个容器在同一个网络中
   - 检查Label Studio是否成功启动
   - 验证API密钥是否正确

3. **GPU不可用**：
   - 确保NVIDIA驱动和容器工具包已正确安装
   - 检查GPU是否被其他进程占用
   - 尝试在`.env`文件中设置`CUDA_VISIBLE_DEVICES=-1`切换到CPU模式

4. **数据导入/导出问题**：
   - 检查共享目录的权限
   - 确保目录路径正确
   - 检查磁盘空间是否充足

## 扩展功能

系统可以根据需要进行以下扩展：

1. **支持更多格式的标注数据**：
   - 修改导入/导出代码以支持其他标注格式

2. **调整主动学习策略**：
   - 实现不确定性采样或其他主动学习策略
   - 修改`active_learning.py`中的相关代码

3. **添加Web界面**：
   - 创建额外的容器提供监控和控制界面

4. **与CI/CD集成**：
   - 添加自动测试和部署脚本

5. **多模型支持**：
   - 扩展代码以支持多个不同的YOLO模型

## 维护和更新

要更新系统组件：

1. **更新Label Studio**：
   ```bash
   docker-compose pull label-studio
   docker-compose up -d
   ```

2. **更新YOLO服务**：
   - 修改代码或依赖
   - 重新构建容器：
   ```bash
   docker-compose build yolo-service
   docker-compose up -d
   ```

## 安全注意事项

1. 不要在公共网络上暴露Label Studio接口
2. 定期更新容器和依赖
3. 不要在环境变量中存储敏感凭据
4. 为共享卷设置适当的权限