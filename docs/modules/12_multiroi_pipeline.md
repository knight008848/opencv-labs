# 模块 12：多 ROI 视野分割与数据管道

> Day 26-27 · Week 4 · 对应 domain_map.md 概念 #7, #12, #20 + 综合
> 在地图中的位置：最上层 — 将前三周所有技能串联为数据管道
> 🎯 通向最终项目：这一步是管道的核心——多视角特征提取 + 结构化输出

---

## 学习目标

- 理解"多 ROI 模拟多相机"的设计思想
- 能对单帧切分多个 ROI，每个 ROI 独立完成完整的分析 Pipeline
- 掌握帧间特征对齐策略（同一 ROI 的跨帧匹配）
- 能生成结构化输出（JSON 语义描述 + NPZ 特征向量）
- 理解这一技能在数据管道中的位置：**Step 2-3 — 特征提取 + 数据打包**

---

## 概念 A：多 ROI — 单相机模拟多视角

### 一句话解释
用 NumPy 切片把一帧分成多个 ROI（感兴趣区域），每个 ROI 模拟一个"虚拟相机"——全局相机、操作台特写、传送带入口。

### 生活类比
你坐在监控中心，面前有 3 块屏幕：左边是大厅全景（ROI 1：全局），中间是柜台特写（ROI 2：操作区），右边是入口（ROI 3：传送带入口）。但其实——所有画面都来自**同一台**天花板的 4K 摄像头。管理员只是把 4K 画面的不同区域"裁剪放大"到不同屏幕上。这就是多 ROI 的核心思想。

### 技术解释
```python
roi_config = {
    "global_view":  (0, 0, img_w, img_h),           # 全图
    "work_area":    (100, 50, 300, 250),             # 操作区
    "inlet":        (400, 300, 200, 150),            # 入口
}

for roi_name, (x, y, w, h) in roi_config.items():
    roi_frame = frame[y:y+h, x:x+w]
    # 对 roi_frame 做完整的分析 Pipeline
    features = analyze_roi(roi_frame)
    results[roi_name] = features
```
每个 ROI 是一个独立的小图，可以各自设置不同的分析策略——操作区看颜色变化、入口看是否有新物体进入。

### 真实案例
工业产线上，一个 8K 线扫描相机覆盖整条传送带。软件把画面分成 6 个 ROI：进料口 → 清洗 → 烘干 → 喷涂 → 质检 → 包装。每个 ROI 跑不同的视觉算法（颜色检查/缺陷检测/尺寸测量），但数据源是同一个相机。

### 练习（3 分钟）
读取一张桌面照片，手动定义 3 个 ROI（左上/右下/中央各一个），用不同颜色矩形框在原图上标注 ROI 边界并保存。
```python
# 你的代码...
```

---

## 概念 B：每 ROI 独立分析 Pipeline

### 一句话解释
对每个 ROI 跑一遍前三周学过的完整流程：灰度→滤波→边缘/二值化→轮廓→几何分析——每个 ROI 输出自己的"物体清单"。

### 生活类比
三个质检员（三个 ROI）同时站在传送带的不同位置。每个质检员手里都有同样的工具包（我们在 Week 1-3 学的所有函数）。他们各自独立工作，只汇报自己负责区域的结果。主控室（管道）收集三份报告，汇总。

### 技术解释
```python
def analyze_roi(roi_frame):
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # 方法 A：颜色过滤
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # 方法 B：边缘+轮廓
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"]/M["m00"]) if M["m00"]>0 else 0
            cy = int(M["m01"]/M["m00"]) if M["m00"]>0 else 0
            objects.append({"bbox":[x,y,w,h], "centroid":[cx,cy],
                          "area":cv2.contourArea(cnt)})
    return objects
```

### 真实案例
一个 4K 监控画面中，同时检测：停车位有无车辆（ROI1）、行人是否越线（ROI2）、红绿灯状态（ROI3）。每个 ROI 的检测逻辑完全不同，但共享同一个输入帧。

### 练习（3 分钟）
对你录的桌面视频的一帧，定义 3 个 ROI，每个 ROI 调用你之前写的任意一个检测函数（如颜色过滤、Canny 边缘、轮廓检测），输出每个 ROI 检测到的物体数。
```python
# 你的代码...
```

---

## 概念 C：帧间数据对齐

### 一句话解释
同一 ROI 在连续帧中检测到的物体需要"对号入座"——帧 N 的物体 A 在帧 N+1 中还是物体 A（可能位置稍有变化），通过距离匹配建立跨帧关联。

### 生活类比
你是一个体育记者，在篮球场边拍照。第 1 秒你拍了全场（帧 1），标记了 10 个球员的位置。第 2 秒你又拍了一张（帧 2），球员位置全变了。你怎么知道第 2 秒照片里的"红队 7 号"是第 1 秒照片里的同一个人？——看他不可能在 1 秒内从球场这头跑到那头，所以"找最近的那个红队 7 号"就行。

### 技术解释
```python
def match_objects(prev_objects, curr_objects, max_dist=50):
    """Match objects between frames using nearest-centroid."""
    matches = {}
    matched_curr = set()
    
    for i, prev in enumerate(prev_objects):
        best_j, best_dist = None, float('inf')
        for j, curr in enumerate(curr_objects):
            if j in matched_curr: continue
            dist = np.linalg.norm(
                np.array(prev["centroid"]) - np.array(curr["centroid"])
            )
            if dist < best_dist:
                best_dist, best_j = dist, j
        
        if best_dist < max_dist:
            matches[i] = best_j
            matched_curr.add(best_j)
    
    return matches  # {prev_idx: curr_idx}
```
`max_dist` 是核心参数——太大容易错配，太小容易断配。一般取物体尺寸的 2-3 倍。

### 真实案例
自动驾驶多目标追踪（MOT）：前帧检测到 5 个行人 + 3 辆车 → 本帧检测到 5 个行人 + 2 辆车 → 匹配后确认"那辆车已经驶出视野（消失），行人都在（持续追踪）"。

### 练习（3 分钟）
从 Day 25 的运动追踪代码中，把"颜色相同"作为辅助特征加入匹配判断（同颜色的物体更可能是同一个）。
```python
# 你的代码...
```

---

## 概念 D：结构化数据打包

### 一句话解释
把所有帧、所有 ROI 的分析结果统一格式，输出 JSON（人类可读的语义层）+ NPZ（机器友好的特征向量层）。

### 生活类比
工厂一天的质检数据：质检员用自然语言写了一份报告（JSON："下午 3 点 15 分，3 号工位检测到红色工件，面积 452 px²"），同时把工件的 X 光照片存进数据库（NPZ：256 维的特征向量）。前者给人看，后者给 AI 模型用。

### 技术解释
```python
import json, numpy as np

output = {
    "meta": {"video": "my_video.mp4", "fps": 30, "total_frames": 900},
    "frames": [
        {
            "frame_id": 0,
            "timestamp": 0.0,
            "rois": {
                "global_view": {
                    "objects": [
                        {"id": 1, "bbox": [100,50,80,60], "centroid": [140,80],
                         "area": 4800, "color": "red"}
                    ]
                },
                "work_area": {"objects": [...]},
                "inlet": {"objects": []}
            }
        },
        # ... more frames
    ]
}

with open("output.json", "w") as f:
    json.dump(output, f, indent=2)

# 特征向量单独存
np.savez("features.npz",
    frame_0_global=np.array([[140,80,4800,0,0,255]]),
    frame_1_global=np.array([[142,82,4780,0,0,255]]),
)
```
JSON 存语义信息（什么颜色、什么形状、在哪个位置），NPZ 存"向量化后的观察"（模型可以直接输入的数字）。

### 真实案例
VLA 数据集的标注格式：每个 episode 是一个 JSON 文件，记录每一帧的观察（物体位置、机械臂状态）+ 对应的动作标签。NPZ 存视觉嵌入向量（224×224 图像经过 ResNet 的 512 维特征），供模型训练时快速加载。

### 练习（3 分钟）
把 Day 18（轮廓分析器）的输出结果改写成 JSON 格式，保存并验证能用 `json.load` 正确读回。
```python
# 你的代码...
```

---

## Day 26 交付任务

**文件：** `experiments/day_26_multiroi.py`

**任务：** 多 ROI 分析器。输入一段 MP4 视频的一帧（取第 100 帧，或用户可指定帧号），完成：
1. 手动定义 3 个 ROI 区域：`"全景"`（全图）、`"中央操作区"`（画面中央 60%）、`"左上角传送带入口"`（左上 25%）
2. 在原图上用 3 种不同颜色的矩形框标注 ROI 边界
3. 每个 ROI 独立运行完整分析：灰度→高斯模糊→Canny→findContours→过滤→boundingRect
4. 在每个 ROI 内标注检测到的物体（绿色轮廓 + ID 标签）
5. 生成并排对比图：原图+ROI标注 | ROI1分析 | ROI2分析 | ROI3分析（四合一）
6. 终端打印每个 ROI 的检测结果表格

**验收标准：**
- [ ] 3 个 ROI 的颜色标注清晰不重叠
- [ ] 每个 ROI 的物体检测逻辑一致（用了同样的 Pipeline 函数）
- [ ] 四合一对比图标签完整
- [ ] 代码中 ROI 分析函数只有一个，通过参数切换 ROI

---

## Day 27 交付任务

**文件：** `experiments/day_27_pipeline.py`

**任务：** 视频数据管道 v0.1。输入一段 MP4，输出结构化数据：
1. **帧提取**：每 N 帧取 1 帧（N 可配置，默认 5）
2. **多 ROI 分析**：每帧切分 ROI，每个 ROI 跑分析 Pipeline
3. **帧间追踪**：同一 ROI 的物体跨帧匹配（质心距离 < 50px）
4. **结构化输出**：
   - `data/processed/day_27/output.json`：逐帧逐 ROI 的检测结果
   - `data/processed/day_27/features.npz`：每帧每个物体的特征向量 `[cx, cy, area, b, g, r]`
   - `data/processed/day_27/summary.txt`：总帧数、总物体数、物体出现/消失时间线
5. **可视化**：生成一段输出视频，显示检测框 + 物体追踪线

**验收标准：**
- [ ] JSON 格式正确，可用 `json.load` 读取
- [ ] NPZ 格式正确，可用 `np.load` 读取
- [ ] 至少 3 帧有完整的 ROI 分析结果
- [ ] 帧间追踪至少有一对成功匹配
- [ ] 可视化视频中轨迹线清晰

---

## 测验

### Q1（选择）
多 ROI 策略的核心优势是什么？
- A. 提高图像分辨率
- B. 用单相机模拟多视角，对不同区域用不同分析策略
- C. 减少 CPU 使用
- D. 让视频播放更流畅

### Q2（填空）
帧间物体匹配的 `max_dist` 参数，如果设得太大会导致 ____，设得太小会导致 ____。一般取物体尺寸的 ____ 倍。

### Q3（选择）
以下哪个做法是"结构化输出"的正确实践？
- A. 把所有数据存为一个大字符串
- B. JSON 存语义信息 + NPZ 存特征向量
- C. 只在屏幕上打印结果
- D. 把特征向量写入 JSON 的字符串字段

### Q4（判断）
`np.savez` 可以一次保存多个 NumPy 数组到一个文件，每个数组有一个键名。

### Q5（简答）
在具身数据管道中，为什么要把"语义信息（JSON）"和"特征向量（NPZ）"分开存储？各有什么用途？

---

## 复盘问题

1. 你的 3 个 ROI 在 Day 26 的分析结果差异大吗？哪个 ROI 最适合检测物体？为什么？
2. 帧间追踪的 max_dist 你是怎么选的？有没有出现"断配"或"错配"的情况？
3. Day 27 的管道在处理 30 秒视频时大概花了多长时间？最慢的环节是哪个？有加速思路吗？

---

## 参考答案

Q1: B — 多 ROI = 不同区域不同策略，模拟多相机分工

Q2: 错配（不同物体被当成同一个）；断配（同一物体被当成消失了）；2-3 倍

Q3: B — JSON 语义 + NPZ 向量，各取所长

Q4: 正确 — `np.savez('file.npz', a=arr1, b=arr2)` 存多个数组

Q5: JSON 给人读和做逻辑判断（如"检测到红色物体"），NPZ 给模型训练/推理用（如 512 维特征向量直接输入 VLA 的视觉编码器）。分开存储可以各自独立更新——换一个特征提取算法时只更新 NPZ，不改 JSON 结构。
