# Kyoto NMT - Gradio 最小可运行Demo

## 1) 准备软件（Windows）
- Miniconda（或 Anaconda）
- Visual Studio Code（安装 Microsoft 出的 **Python** 扩展）

## 2) 新建并激活conda环境
```bat
conda create -n kyoto-nmt python=3.11 -y
conda activate kyoto-nmt
```

## 3) 安装PyTorch（两选一）
- **有NVIDIA GPU：**
```bat
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
- **没有GPU / 不想用GPU：**
```bat
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

## 4) 安装Python依赖（pip）
```bat
pip install transformers sentencepiece gradio
```

## 5) 打开项目 & 运行
- VS Code: File -> Open Folder... 选择此项目文件夹
- 右下角选择解释器：`Conda (kyoto-nmt)`
- 打开 `app/simple_demo.py`
- 方式A：VS Code 右上角“Run Python File”
- 方式B：终端运行：
```bat
python app\simple_demo.py
```

浏览器打开：http://127.0.0.1:7860 ，输入日文，点击“翻译”。

---

## 6) 常见问题
- 首次运行会从 Hugging Face 下载模型，如网络慢请耐心等候。
- 若 `sentencepiece` 缺失：`pip install sentencepiece`（上面已经包含）。
- 端口被占用：修改 `simple_demo.py` 末尾的 `server_port`。