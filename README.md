# opencv-labs

OpenCV 30 天掌握计划 — 每天一个可交付脚本，最终完成可展示的实战项目。

## 这个仓库是什么

不是文档合集，而是一个**每日可交付**的编码训练营。每天产出一个可运行的 Python 脚本，四周完成一个智能文档扫描仪。

## 环境

```bash
source ~/miniforge3/bin/activate pydata
python -c "import cv2; print(cv2.__version__)"  # 4.13.0
```

## 快速开始

```
1. 阅读 CLAUDE.md      → 了解教学规则和流程
2. 阅读 domain_map.md   → 建立 OpenCV 知识全景图
3. 阅读 30_day_roadmap  → 看 30 天路线图
4. 开始 Day 1           → docs/modules/01_pixel_color.md
```

## 每日流程（60-90 分钟）

```
1. 阅读当日模块                     — 10 min
2. 完成每个概念的 3 分钟练习         — 15 min
3. 完成当日交付任务（写代码）        — 25 min
4. 做测验 + 对照验收标准自检         — 10 min
5. 更新 progress.md                 — 5 min
6. 回答复盘三问                     — 5 min
```

## 四周路线

| 周 | 天数 | 主题 | 阶段测试 | 核心技能 |
|:---|:---|:---|:---|:---|
| Week 1 | Day 1-7 | 图像基石 | Day 7 | 像素、色彩、ROI、直方图、HSV |
| Week 2 | Day 8-14 | 图像变换 | Day 14 | 缩放、仿射、透视、滤波、边缘 |
| Week 3 | Day 15-21 | 图像分析 | Day 21 | 二值化、形态学、轮廓、霍夫 |
| Week 4 | Day 22-30 | 进阶+项目 | Day 28 | 特征、视频、相机标定、项目实战 |

## 最终项目

**智能文档扫描仪**：手机拍歪斜文档 → 自动矫正为正视角 + 干净黑白扫描件。

用到：透视变换 + 边缘检测 + 轮廓 + 二值化

## 项目结构

```
├── CLAUDE.md               # 教学规则（必须先读）
├── domain_map.md            # OpenCV 知识全景图
├── progress.md              # 学习进度追踪（每日更新）
├── README.md                # 本文件
├── docs/
│   ├── 30_day_roadmap.md    # 30 天路线总览
│   └── modules/             # 10 个学习模块
│       ├── 01_pixel_color.md
│       ├── 02_roi_histogram.md
│       ├── ...
│       └── 10_features_matching.md
├── experiments/             # 每日脚本（按 day_XX 命名）
├── data/raw/                # 原始素材
├── data/processed/          # 处理结果
└── src/utils.py             # 可复用工具
```

## 重要规则

1. **每天必须有可交付成果** — 一个 `.py` 文件
2. **验收标准不通过 = 当天不算完成**
3. **每周一次阶段测试** — Day 7/14/21/28
4. **错题要分类** — [C]不懂概念 / [A]不会应用 / [E]表达不清 / [K]知识混淆
5. **根据薄弱点调整计划** — 正确率 < 60% 安排补习日
6. **最终项目贯穿全程** — 每学一个技能都关联到最终项目
