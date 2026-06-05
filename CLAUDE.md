# opencv-labs

OpenCV 学习项目（从 embodied-ai 迁移）。

## 环境

Python 环境使用 miniforge3 的 `pydata` conda 环境，已安装 OpenCV 4.13.0。

所有 Python 命令必须在 `pydata` 环境中运行：

```bash
source ~/miniforge3/bin/activate pydata
```

## 项目结构

```
data/raw/       - 原始素材（视频、图片）
data/processed/ - 处理后的数据
notebooks/      - Jupyter Notebook 练习
experiments/    - 实验脚本
models/         - 模型文件
src/            - 可复用工具代码
docs/           - 文档
```


# 个人偏好

- 回复使用中文
- 代码注释使用英文
- 优先使用函数式编程风格