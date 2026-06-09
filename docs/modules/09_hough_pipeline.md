# 模块 9：霍夫变换与综合 Pipeline

> Day 19-20 · Week 3 · 对应 domain_map.md 概念 #12 补充
> 在地图中的位置：轮廓分析之上 → 检测直线和圆 + 串联所学技能

---

## 学习目标

- 理解霍夫变换的原理——从图像空间"投票"到参数空间
- 掌握霍夫直线检测和圆检测的调参
- 能将二值化→形态学→轮廓→霍夫串联成完整检测 Pipeline
- 理解 Pipeline 设计原则：模块化、可测试、有中间结果

---

## 概念 A：霍夫直线检测

### 一句话解释
霍夫变换不用"连点成线"的方法找直线，而是让每个边缘像素"投票"——"我可能属于斜率为 m、截距为 b 的直线"——票数最多的 (m, b) 就是图中的直线。

### 生活类比
班里选班长。不直接问"谁是班长"，而是让每个人投票给最合适的候选人。某个候选人票数远超其他人 → 当选。霍夫直线检测同理——每个边缘点不是直接说"我在这条线上"，而是投票给所有"可能经过我的直线"。票数最高的参数组合就是图中的真实直线。

### 技术解释
用极坐标 (ρ, θ) 而非斜截式 (m, b) 来避免垂直线（斜率无穷大）问题。每条直线表示为 `ρ = x·cosθ + y·sinθ`。`HoughLinesP` 返回线段的端点坐标（而非整条无限直线），更实用。

```python
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                        minLineLength=50, maxLineGap=10)
# threshold: 最少投票数（越高=越严格）
# minLineLength: 最短线段长度（过滤短线）
# maxLineGap: 线段间最大间隙（间隙小于此值则连接）
```

### 真实案例
自动驾驶车道线检测：Canny 边缘 → HoughLinesP 找直线 → 按斜率和位置分组（左右车道线）。车道线是图中最显著的长直线，Hough 投票数最高。

### 练习（3 分钟）
创建一张含多条直线的图（如网格纸），用 HoughLinesP 检测并画出所有直线（每条线一个颜色）。
```python
# 你的代码...
```

---

## 概念 B：霍夫圆检测

### 一句话解释
霍夫圆检测在 (x, y, r) 三维参数空间中投票——每个边缘点投票给所有可能经过它的圆心和半径。

### 生活类比
找直线是 2D 投票（斜率和截距），找圆是 3D 投票（圆心 x，圆心 y，半径 r）。就像不是选班长，而是选"班长+副班长+学习委员"三人组——组合投票，票数最高的三人组当选。

### 技术解释
`HoughCircles` 内部使用"梯度法"——先计算每个边缘点的梯度方向，圆心应位于梯度方向上（大幅减少搜索空间）。关键参数：`dp`（累加器分辨率）、`minDist`（圆心最小间距）、`param1`（Canny 高阈值）、`param2`（圆心累加器阈值，越小=越多假阳性）。

```python
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=100, param2=30, minRadius=10, maxRadius=200)
```

### 真实案例
工业零件计数：传送带上有一堆圆形垫圈，HoughCircles 检测每个垫圈的位置和大小，自动计数。相比轮廓法，霍夫圆对重叠和部分遮挡的圆形鲁棒性更好。

### 练习（3 分钟）
找一张含硬币的照片（或画几个圆），用 HoughCircles 检测所有圆，调节 `param2` 观察"检测数 vs 误检数"的 trade-off。
```python
# 你的代码...
```

---

## 概念 C：Pipeline 设计原则

### 一句话解释
Pipeline = 将多个处理步骤串联，前一步的输出是后一步的输入——每一步只做一件事、可独立测试。

### 生活类比
工厂流水线——第一个工位放零件，第二个焊接，第三个喷漆，第四个质检。你不会让一个工人做完所有事。某个工位出问题，只修那一个就行，不用整条线停掉。每个工位的输入输出都有明确规格。

### 技术解释
好的 Pipeline 设计：
- 每步一个独立函数，输入输出类型明确
- 每步保存中间结果（方便调试）
- 参数集中在配置文件或函数参数中
- 错误处理：输入为空、格式不对、处理失败都要有降级方案

```python
def pipeline(image_path, config):
    # Step 1: Load
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"Cannot load {image_path}")
    # Step 2: Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config["blur_ksize"], 0)
    # Step 3: Edge detect
    edges = cv2.Canny(blurred, config["canny_t1"], config["canny_t2"])
    # Step 4: Morphology
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, config["kernel"])
    # Step 5: Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Step 6: Filter + Analyze
    results = [analyze(cnt) for cnt in contours if cv2.contourArea(cnt) > config["min_area"]]
    return results
```

### 真实案例
任何工业视觉系统都是 Pipeline：图片采集→预处理→检测→分类→决策→执行。每个环节由不同团队开发和维护。

### 练习（3 分钟）
把你 Day 17 或 Day 18 的代码重构为一个 Pipeline 函数，每个步骤一个独立函数，通过 dict 传参。
```python
# 你的代码...
```

---

## Day 19 交付任务

**文件：** `experiments/day_19_hough.py`

**任务：** 霍夫检测实验室。输入图片（建议：棋盘格、含硬币的桌面），完成：
1. 霍夫直线检测：画出所有检测到的线段（长度 > 30 像素）
2. 霍夫圆检测：画出所有检测到的圆（圆心 + 圆周）
3. 并排对比：原图 | 直线检测结果 | 圆检测结果
4. 终端打印检测到的直线数和圆数
5. 提供 Trackbar 调节关键参数（如直线 threshold、圆 param2）

**验收标准：**
- [ ] 直线/圆检测无明显漏检或误检
- [ ] Trackbar 实时更新检测结果
- [ ] 直线和圆的绘制颜色区分清楚

---

## Day 20 交付任务

**文件：** `experiments/day_20_pipeline.py`

**任务：** 完整目标检测 Pipeline。设计一个统一的 Pipeline 函数，输入图片路径和配置字典，输出检测结果：
1. Pipeline 步骤：`加载 → 灰度 → 高斯模糊 → Canny → 形态学闭运算 → findContours → 过滤 → 几何分析`
2. 每步保存中间结果到 `data/processed/day_20/steps/`（调试用）
3. 最终输出：标注图 + JSON 结果文件（每个物体的位置、面积、形状分类）
4. 配置文件用 Python dict（不使用 YAML 文件以减少依赖）

**验收标准：**
- [ ] 6 个步骤的中继结果全部可查看
- [ ] JSON 输出格式规范，可用 `json.load` 读取
- [ ] 对至少 3 张不同图片运行成功
- [ ] 每步有 try/except 处理（预处理失败、轮廓为空等）

---

## 测验

### Q1（选择）
霍夫变换用极坐标 (ρ, θ) 而非斜截式 (m, b) 的原因：
- A. 极坐标计算更快
- B. 斜截式无法表示垂直线（m=∞）
- C. 极坐标更精准
- D. 没有特殊原因

### Q2（填空）
`HoughLinesP` 的 `threshold` 参数含义是 ____，值越大检测到的直线越 ____。`minLineLength` 用于过滤 ____。

### Q3（选择）
关于 Pipeline 设计，以下哪种做法最差：
- A. 每个步骤独立函数
- B. 所有步骤写在一个大函数里
- C. 参数通过 dict 传递
- D. 保存中间结果以便调试

### Q4（判断）
霍夫圆检测比轮廓法更适合检测重叠的圆形物体。

### Q5（简答）
在目标检测 Pipeline 中，如果二值化输出的白色区域太多（背景也被当成了前景），后续的轮廓分析会怎样？应该如何在前几步预防？

---

## 复盘问题

1. 霍夫直线检测的 `threshold` 和 Canny 的 `threshold` 都叫 threshold——但它们是完全不同的概念。你能区分吗？
2. Day 20 的 Pipeline 在你测试的 3 张图片上都成功了吗？如果某张图失败，是哪一步出了问题？
3. 这个 Pipeline 如果要应用到实时视频（30 FPS），你估计最慢的步骤是什么？怎么优化？

---

## 参考答案

Q1: B — 斜截式 m (斜率) 在垂直线时无穷大，计算机无法表示

Q2: 最少投票数（交点数）；少；过短的线段

Q3: B — 单函数难以调试、测试、复用，违反单一职责原则

Q4: 正确 — 霍夫圆基于梯度投票，即使部分轮廓被遮挡也能检测到圆

Q5: 二值化太"松" → 太多白色区域 → findContours 找到大量无关轮廓 → 过滤步骤压力大/可能漏过滤。预防：调高 Canny 阈值、增大形态学核、或在前端改善光照。
