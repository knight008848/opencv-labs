# 模块 8：轮廓提取与几何分析

> Day 17-18 · Week 3 · 对应 domain_map.md 概念 #12
> 在地图中的位置：形态学之上 → 从二值图中提取物体边界 + 测量物体属性

---

## 学习目标

- 掌握 `findContours` 的检索模式和层级关系
- 理解 `CHAIN_APPROX_SIMPLE` 的压缩原理
- 能计算轮廓的面积、周长、质心、边界框、圆度等几何属性
- 掌握 `approxPolyDP` 做轮廓近似和形状分类
- 能独立完成"检测图中物体 + 标注属性"的完整流程

---

## 概念 A：轮廓提取

### 一句话解释
`findContours` 在二值图中追踪所有白色区域的边界，返回每个闭合边界的像素坐标列表。

### 生活类比
你在白纸上画了几个封闭的圈——有些圈里还套了小圈（像"回"字）。`findContours` = 你拿一支笔沿着所有圈的边界走一遍，记录下沿途的坐标。`RETR_EXTERNAL` = 只看最外层的圈（忽略里面的）。`RETR_TREE` = 记下所有圈，并且注明"这个圈在那个圈里面"（父子关系）。

### 技术解释
输入必须是二值图（前景=白色=255）。`CHAIN_APPROX_SIMPLE` 压缩水平/垂直/对角线段——只保留端点（如矩形的 4 个角），而非边上的每个像素。返回 `contours` 列表，每个元素是 `[[[x1,y1]], [[x2,y2]], ...]` 格式。

```python
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# hierarchy[i] = [next, previous, first_child, parent] — 4 个索引
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # -1 = 画所有轮廓
```

### 真实案例
药片缺陷检测：传送带上拍照后二值化 → `findContours` 找到每个药片的轮廓。如果某个药片的轮廓面积明显小于其他药片（缺角），标记为次品。

### 练习（3 分钟）
绘制一张含圆形、矩形、三角形的二值图，用 `findContours` 检测并 `drawContours` 画出所有轮廓（不同颜色）。打印检测到的轮廓数量和层级信息。
```python
# 你的代码...
```

---

## 概念 B：轮廓几何属性

### 一句话解释
拿到轮廓（一组点）后，可以计算它的面积、周长、质心、外接矩形、外接圆、"圆度"等——用数字描述一个形状。

### 生活类比
你在地上用粉笔画了一个不规则的圈。想知道它有多大（面积）——数圈里的地砖。圈有多长的边（周长）——拿卷尺沿着边界量。刚好能包住它的最小纸箱（bounding rect）有多大——拿一个纸箱试着套。这个圈有多"圆"（圆度）——如果圆的圆度=1，你的圈的圆度肯定小于 1。

### 技术解释
- `contourArea(cnt)` —— 面积，带符号（顺时针为负），用格林公式计算
- `arcLength(cnt, True)` —— 周长，True=闭合轮廓
- `boundingRect(cnt)` —— 轴对齐外接矩形，返回 `(x, y, w, h)`
- `minAreaRect(cnt)` —— 最小面积旋转矩形，返回 `(center, size, angle)`
- `cv2.moments(cnt)` —— 图像矩，`m10/m00`=质心 x，`m01/m00`=质心 y
- 圆度 = `4π × 面积 / 周长²` —— 完美圆=1.0，越不规则越小

```python
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
x, y, w, h = cv2.boundingRect(cnt)
M = cv2.moments(cnt)
cx = int(M["m10"] / M["m00"])  # 质心 x
cy = int(M["m01"] / M["m00"])  # 质心 y
circularity = 4 * np.pi * area / (perimeter * perimeter)
```

### 真实案例
硬币分类器：检测所有圆形物体，通过 `minAreaRect` 的尺寸和 `contourArea` 的面积估计硬币面值（1 元 > 5 角 > 1 角），计算总金额。

### 练习（3 分钟）
对上一步找到的每个轮廓，计算并打印面积、周长、质心、圆度。过滤掉面积 < 50 的噪点轮廓。
```python
# 你的代码...
```

---

## 概念 C：轮廓近似与形状分类

### 一句话解释
`approxPolyDP` 用更少的顶点近似轮廓——epsilon 控制精度。近似的顶点数可以用来判断形状：3 个顶点=三角形，4 个=矩形，>8 个=圆。

### 生活类比
你要用最少的小旗子在地面上标注一个椭圆形的操场边界。epsilon 越小 = 旗子越多越贴合椭圆。epsilon 越大 = 旗子越少（如只用 8 根旗子标出大致形状），但边界看起来很粗糙。如果只需要判断"这是个圆还是方"，4 根旗子和 40 根旗子的判断结果是一样的。

### 技术解释
`approxPolyDP(cnt, epsilon, True)` 使用 Douglas-Peucker 算法。epsilon = 原轮廓到近似轮廓的最大允许距离，通常用 `0.02 * 周长` 作为起点。近似后的轮廓是一个顶点列表，数顶点就是数边数。

```python
epsilon = 0.02 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
vertices = len(approx)
if vertices == 3: shape = "Triangle"
elif vertices == 4: shape = "Rectangle"
elif vertices > 8: shape = "Circle"
```

### 真实案例
交通标志识别：摄像头拍到圆形/三角形/矩形标志 → `approxPolyDP` 判断形状 → 圆形=限速/禁令、三角形=警告、矩形=指示。形状分类后再做具体的文字/图案识别。

### 练习（3 分钟）
对你画的三角形、矩形、圆形分别做 `approxPolyDP`，验证顶点数是否正确对应了形状。
```python
# 你的代码...
```

---

## Day 17 交付任务

**文件：** `experiments/day_17_contours.py`

**任务：** 物体检测标注器。输入含多个物体的照片（手机拍桌面上的钥匙、硬币、笔等），输出标注图：
1. 完整的 Pipeline：灰度 → 高斯模糊 → Canny → findContours
2. 过滤面积 < 500 的噪点
3. 用不同颜色画出每个物体的轮廓（颜色循环）
4. 在每个物体旁标注：ID 编号 + 面积
5. 保存到 `data/processed/day_17_labeled.jpg`

**验收标准：**
- [ ] 所有显著物体都有轮廓和 ID（漏检 < 20%）
- [ ] 颜色区分明显（相邻物体颜色不同）
- [ ] 标注文字不遮挡物体

---

## Day 18 交付任务

**文件：** `experiments/day_18_analysis.py`

**任务：** 几何属性分析器。扩展 Day 17 的代码：
1. 对每个物体计算：面积、周长、最小外接矩形（旋转+非旋转）、圆度、长宽比
2. 用 `approxPolyDP` 判断形状类型（三角形/矩形/圆形/不规则）
3. 绘制最小外接旋转矩形（绿色）和轴对齐矩形（蓝色）
4. 生成 CSV 报告表：`id, shape, area, perimeter, circularity, aspect_ratio, cx, cy`
5. 保存标注图和 CSV 到 `data/processed/day_18/`

**验收标准：**
- [ ] 形状分类正确率 > 80%
- [ ] CSV 数据与图片标注一致
- [ ] 旋转矩形真正"贴合"倾斜物体（而不是水平的）

---

## 测验

### Q1（选择）
`findContours(binary, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` 中，`RETR_EXTERNAL` 返回：
- A. 所有轮廓包括内层轮廓
- B. 只返回最外层轮廓
- C. 返回所有轮廓但不包含层级信息
- D. 返回层级树

### Q2（填空）
圆度（circularity）的公式是 ____。完美圆的圆度 = ____。正方形的圆度 ≈ ____（计算：面积=1, 周长=4 → 4π×1/16）。

### Q3（选择）
`approxPolyDP` 中 epsilon 值越大，结果如何变化？
- A. 近似轮廓的顶点越多
- B. 近似轮廓的顶点越少，形状越粗糙
- C. 近似更精确
- D. epsilon 不影响顶点数

### Q4（判断）
`boundingRect` 返回的是可以旋转的最小矩形。

### Q5（代码补全）
```python
# 计算轮廓的质心
M = cv2.____(cnt)
cx = int(M["____"] / M["____"])
cy = int(M["____"] / M["____"])
```

---

## 复盘问题

1. 如果两个物体在图中重叠了，`findContours` 能分开它们吗？如果不能，你有什么办法？
2. 圆度能完美区分圆形和正方形吗？试试看——正方形的圆度理论值是多少？
3. `approxPolyDP` 判断形状的策略在什么情况下会失效？（提示：物体被遮挡、透视变形）

---

## 参考答案

Q1: B — RETR_EXTERNAL 只取最外层轮廓

Q2: `4π×Area / Perimeter²`; 1.0; ≈0.785 (=π/4)

Q3: B — epsilon 越大近似越粗糙，顶点越少

Q4: 错误 — `boundingRect` 是轴对齐的（不旋转）；`minAreaRect` 才可以旋转

Q5: `moments`, `m10`, `m00`, `m01`, `m00`
