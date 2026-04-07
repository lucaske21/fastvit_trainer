# 使用 NVIDIA PyTorch 官方鏡像作為基礎鏡像
FROM nvcr.io/nvidia/pytorch:23.03-py3

# 設置工作目錄
WORKDIR /workspace

# 設置環境變量，避免交互式對話
ENV DEBIAN_FRONTEND=noninteractive

# 更新系統並安裝必要的依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製源代碼到容器中
COPY . .

# 設置 PYTHONPATH
ENV PYTHONPATH=/workspace

# 暴露端口（如果需要使用 Tensorboard）
EXPOSE 6006

# 默認命令
CMD ["bash"]
