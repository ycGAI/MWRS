#!/usr/bin/env python3
import os
import sys

# 读取原始文件
filename = sys.argv[1]
with open(filename, 'r') as f:
    content = f.read()

# 修改PROJECT_ID的处理方式
if "PROJECT_ID = int(os.environ.get(\"PROJECT_ID\"" in content:
    modified_content = content.replace(
        "PROJECT_ID = int(os.environ.get(\"PROJECT_ID\", 1))",
        """# 确保环境变量正确处理
try:
    PROJECT_ID = int(os.environ.get("PROJECT_ID", 1))
    print(f"Successfully initialized PROJECT_ID={PROJECT_ID}")
except Exception as e:
    print(f"Error getting PROJECT_ID from environment, using default: {e}")
    PROJECT_ID = 1"""
    )
    
    # 保存修改后的文件
    with open(filename, 'w') as f:
        f.write(modified_content)
    print(f"Successfully patched {filename}")
else:
    print(f"No need to patch {filename}, PROJECT_ID already fixed")
