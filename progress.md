# 学习进度追踪

> 每天学习结束后更新。这是学习档案，也是我调整教学计划的依据。

## 当前状态

- **开始日期：** 2026-06-09
- **当前天数：** Day 20 / 30
- **当前模块：** 模块 9 概念 C — Pipeline 设计（7 步检测流水线：load→gray→blur→Canny→morph→contours→analyze）
- **完成率：** 67%
- **最终项目：** 具身视觉数据管道（MP4 -> 结构化观察数据）
- **累计编码时间：** ~25 小时

---

## 每日进度

| 日期 | 天数 | 模块 | 练习完成 | 测验正确率 | 耗时 | 备注 |
|:---|:---|:---|:---|:---|:---|:---|
| 06-09 | Day 0 | - | - | - | - | 初始化学习环境 |
| 06-10 | Day 1 | 模块 1 | 3/3 | 60% (3/5) | ~1.5h | 像素矩阵探索 + 测验 |
| 06-11 | Day 2 | 模块 1 | 3/3 | 100% (5/5) | ~1h | 色彩空间转换 + 通道操作 + colormap |
| 06-11 | Day 3 | 模块 2 | 3/3 | 80% (4/5) | ~1h | ROI切片 + 九宫格拼图 + copyMakeBorder |
| 06-15 | Day 4 | 模块 2 | 3/3 | 90% (4.5/5) | ~1h | 直方图计算 + 均衡化 + 曝光评估 |
| 06-16 | Day 5 | 模块 3 | 3/3 | 100% (5/5) | ~1h | addWeighted 混合 + absdiff 差异检测 |
| 06-17 | Day 6 | 模块 3 | 3/3 | - | ~1.5h | HSV inRange 5色过滤 + findContours 物体计数 |
| 06-17 | Day 7 | 阶段测试 1 | - | 86% (6/7) | ~0.5h | 模块 1-3 综合测试 |
| 06-18 | Day 8 | 模块 4 | 1/1 | - | ~1h | letterbox 批量预处理 224x224 |
| 06-18 | Day 9 | 模块 4 | 1/1 | 50% (2.5/5) | ~1.5h | safe_rotate + 金字塔，模块测验 |
| 06-22 | Day 10 | 模块 5 | 1/1 | 80% (4/5) | ~1h | 透视变换4点矫正 + ginput交互 + 逆warp合成测试图 |
| 06-22 | Day 11 | 模块 5 | 1/1 | 83% (5/6) | ~1h | 6合1滤波对比 + Canny边缘保留度 + benchmark耗时排序 |
| 06-23 | Day 12 | 模块 6 | 1/1 | 83% (5/6) | ~1.5h | Sobel/Canny/Laplacian 边缘检测 + Canny 阈值扫描 + 自动 Canny |
| 06-24 | Day 13 | 模块 6 | 1/1 | - | ~1.5h | 组合 Pipeline：中值去噪 -> 高斯平滑 -> Canny -> 最大四边形 -> 透视矫正 |
| 06-24 | Day 14 | 阶段测试 2 | - | 87% (8.7/10) | ~0.5h | 模块 4-6 综合测试 |
| 06-29 | Day 15 | 模块 7 | 2/2 | - | ~2h | 3 速练 + 二值化对比报告 + BINARY/BINARY_INV + 真实图验证 + code review 修复 |
| 07-02 | Day 16 | 模块 7 | 3/3 | 90% (4.5/5) | ~1.5h | 形态学 6 操作对比 + 3 种结构元素 + 合成含噪二值图 + 网格图分析 |
| 07-14 | Day 17 | 模块 8 | 1/1 | - | ~1h | 轮廓提取 Pipeline：灰度→Canny→findContours→面积过滤→彩色标注→debug 网格 |
| 07-16 | Day 18 | 模块 8 | 1/1 | 100% (5/5) | ~1h | 几何属性 + approxPolyDP 形状分类 + CSV 报告 + 模块测验 |
| 07-17 | Day 19 | 模块 9 | 1/1 | 75% (6/8) | ~1.5h | 霍夫直线/圆检测 + 参数 sweep + 并排对比 + 模块测验 |
| 07-22 | Day 20 | 模块 9 | 1/1 | - | ~2h | 7 步检测 Pipeline + 视频帧验证 + 中间步骤可视化 + JSON 报告 |

---

## 每日学习日志

### Day 17 (2026-07-14) — 模块 8 概念 A：轮廓提取

**完成事项：**
- [x] 骨架搭建：load_image / preprocess / find_objects / draw_labeled_objects / build_debug_grid
- [x] find_objects：findContours + RETR_EXTERNAL + CHAIN_APPROX_SIMPLE + area 过滤 + 降序排序
- [x] get_color_palette：HSV 色相均匀采样 → BGR 元组，保证轮廓颜色区分
- [x] draw_labeled_objects：逐轮廓上色 + cv2.moments 质心 + putText 标注 ID/面积
- [x] build_debug_grid：2×2 管线中间结果可视化
- [x] main：PORTABLE 路径 + 真实照片自动缩放 + 全流程接通
- [x] 空轮廓保护（get_color_palette(0) 不会崩溃）
- [x] 跑通 IMG_0701.png：找到 10 个物体，面积 769~2,202 px²

**关键发现：**
- Canny 边缘占比 2.3%，阈值 (50, 150) 对桌面场景适中——没有过度噪点也没漏主要轮廓
- CHAIN_APPROX_SIMPLE 压缩后轮廓点数大幅减少，但 contourArea 精度不受影响
- 路径写死 vs SCRIPT_DIR/PROJECT_DIR 可移植性差异明显——后者在 VM 和本地之间自动适配

**复盘三问：**
1. findContours 返回的是坐标列表还是二值图？坐标列表。每个轮廓是一组 `(x, y)` 点，不是像素掩码。这是"图像处理"和"图像分析"的分水岭。
2. RETR_EXTERNAL vs RETR_TREE 什么时候用？只数物体个数→EXTERNAL；需要知道物体嵌套关系（如在框里的文字）→TREE。
3. 轮廓 Pipeline 和最终项目的关系？数物体+算属性正是最终项目"多 ROI 分窗分析"的核心：每个 ROI 独立做轮廓提取和特征计算。

### Day 18 (2026-07-16) — 模块 8 概念 B+C：轮廓几何属性 + 形状分类

**完成事项：**
- [x] compute_properties：area / arcLength / moments 质心 / boundingRect / minAreaRect / circularity / aspect_ratio
- [x] m00 零保护 + minAreaRect 三层解包陷阱修复
- [x] classify_shape：approxPolyDP 四种形状分类（Triangle / Rectangle / Circle / Irregular）
- [x] draw_boxes：轴对齐矩形（蓝）+ 旋转矩形（绿）双层叠加 + 形状标签
- [x] save_csv：csv.DictWriter 输出属性表（float 四舍五入到 2 位小数）
- [x] build_debug_grid：Panel 4 读取 CSV 并渲染为 monospace 文本表格
- [x] main() 串联完整 Pipeline：合成图 9 个物体全部检出

**关键发现：**
- minAreaRect 返回嵌套三层元组 `((cx,cy),(w,h),angle)`，不是 5 个独立值——Day 18 踩坑
- 合成图（type_test.png）上 9 个物体检出结果：3 Triangle / 2 Rectangle / 2 Circle / 2 Irregular
- 圆度理论值：完美圆=1.0，正方形≈0.785，长条矩形圆度远低于 0.6
- 六角星被 approxPolyDP(epsilon=0.02) 压成 3 个顶点→归类为 Triangle，是 epsilon 过大的典型误分类

**复盘三问：**
1. compute_properties 为什么写 `moments(cnt)` 而不是逐个算？图像矩一次调用能拿到所有矩值（m00、m10、m01…），避免重复遍历轮廓。一次调用、字典取值——更快也更干净。
2. minAreaRect 的三层嵌套怎么解？用 `((cx, cy), (w, h), angle)` 解包，这是最直接的方式。也可以用下标 `rect[0][0]`，但可读性差。
3. approxPolyDP 的形状分类在什么条件下会失效？物体被遮挡导致轮廓残缺，或者透视变形导致矩形成梯形→approx 顶点数变 4 以上→被分错类。epsilon 的 0.02 经验值也需要根据物体大小微调。

### Day 19 (2026-07-17) — 模块 9 概念 A+B：霍夫直线 + 圆检测

**完成事项：**
- [x] detect_lines：HoughLinesP 直线检测（threshold / minLineLength / maxLineGap 调参）
- [x] detect_circles：HoughCircles 圆检测（param2 调参 + dp / minDist / minRadius / maxRadius）
- [x] draw_lines / draw_circles：线段 + 圆周 + 圆心标注
- [x] sweep_line_threshold：threshold=[30,60,100,150] 1×4 网格对比
- [x] sweep_circle_param2：param2=[20,30,40,50] 1×4 网格对比
- [x] build_debug_grid：2×2 灰度/边缘/直线/圆综合展示
- [x] 合成图 type_test.png：检测到 46 条直线 + 9 个圆（param2=30 偏松，有重复检测）

**关键发现：**
- 霍夫直线 vs 轮廓法的本质区别：投票 vs 追踪——前者用参数空间投票找出"最可能存在的几何形状"，后者用连通组件追踪"实际的闭合边界"
- param2=30 检测 9 个圆但包含重复检测（同一个圆被不同半径检测多次），param2=50 会漏检——trade-off 是调参核心
- HoughCircles 输入是灰度图（非边缘图），内部自建 Canny——而 HoughLinesP 输入是边缘图
- 46 条直线中部分线段是连接矩形和三角形的边界线，threshold=60 较好地平衡了"不漏主线"和"不捡噪点"

**复盘三问：**
1. 霍夫直线和轮廓法检测直线的方式有什么不同？轮廓法先找闭合边界再 approxPolyDP 判断边数——前提是直线必须构成封闭图形的边。霍夫直接检测任意方向的非闭合线段——只要边缘点在投票空间中有足够多的共线点。前者"找形状"，后者"找直线"。
2. threshold 在 Hough 和 Canny 中的含义有何区别？Canny 的 threshold 是梯度强度阈值（滞后连接用的）；Hough 的 threshold 是最少投票数（最少有多少个像素共线才算直线）。同名不同义。
3. 霍夫检测和最终项目的关系？最终项目的"帧间物体追踪"环节中，如果场景中有直线特征（如桌面边缘、工件边界），霍夫直线可以作为特征点的补充线索——结合轮廓的质心追踪和霍夫的直线追踪提供更鲁棒的帧间对齐。

### Day 20 (2026-07-22) — 模块 9 概念 C：Pipeline 设计

**完成事项：**
- [x] 7 步检测 Pipeline 骨架：load_image / to_grayscale / apply_blur / detect_edges / morph_close / find_contours / compute_properties + classify_shape
- [x] draw_annotations：轮廓着色 + 轴对齐包围盒 + 质心标签（ID / 面积 / 形状）
- [x] get_color_palette：HSV 色相均匀采样 → BGR 元组（内联实现，消除 Day 17 导入依赖）
- [x] build_debug_grid：2×3 流水线中间结果可视化（Load / Gray / Blur / Canny / Morph / Annotated）
- [x] run_pipeline：7 步串联 + 异常保护 + steps/ 中间结果分包保存
- [x] main：合成图 type_test.png + 真实照片 IMG_0701.png / IMG_0705.png 三图验证
- [x] 视频帧验证：从 rgb_79c1787d6c.mp4 随机抽 4 帧 → 全部跑通（11 / 5 / 112 / 68 / 30 / 38 / 35 物体）
- [x] save_json_report：多图结果汇总 JSON 输出
- [x] 空轮廓保护 + FileNotFoundError 升级 + m00=0 除零保护

**关键发现：**
- 流水线的 7 步顺序有严格依赖：必须先 blur 降噪再 Canny，否则噪声边缘泛滥；必须先 morph 闭合断裂边再 findContours，否则一条轮廓断成好几段
- 视频帧 (1920×1440 @60fps) 需用 max_size=1200 降采样——全分辨率下轮廓数量陡增但大部分是背景纹理
- 形状分类基于 approxPolyDP 顶点数：3→Triangle / 4→Rectangle / ≥8→Circle，但真实场景中许多 "Circle" 的 circularity 仅 0.01~0.14——需后续用 circularity 二次校验

**复盘三问：**
1. Pipeline 的每一步是否可以互换顺序？大部分不行。顺序是：去噪（Blur）→ 边缘（Canny）→ 闭合（Morph）→ 轮廓（Contours）。调换 blur 和 Canny 会多出无数假边缘；去掉 morph 会导致断裂轮廓无法被完整检出。
2. run_pipeline 的接口设计有什么取舍？每步都保存中间结果到文件——方便调试但增加 I/O。最终项目版应该给一个 debug=False 参数跳过中间保存，只保留最终输出。
3. 今天 Pipeline 的架构和最终项目的关系？架构完全一致：帧提取 → 预处理 → 特征提取 → 结构化输出。今天用 Canny+Contours 作为"特征"，最终项目替换成 ROI 分窗 + 帧间追踪即可。

### Day 15 (2026-06-29) — 模块 7 概念 A：三种二值化策略 + BINARY vs BINARY_INV

**完成事项：**
- [x] 速练 1：三种二值化对比（侧光场景）— 全局/Otsu/自适应
- [x] 速练 2：腐蚀再膨胀 = 开运算原理 — 噪点消失 / 矩形瘦了 / 膨胀恢复
- [x] 速练 3：闭运算连断线 + 开运算去噪 — 同一内核、顺序决定效果
- [x] 交付任务 `day_15_threshold.py`：白纸黑字仿真 × 3 种光照 × 6 种方法（BINARY + BINARY_INV）
- [x] `build_comparison_panel()` — 4×2 网格：左 BINARY / 右 BINARY_INV，含黑白像素比标注
- [x] `write_report()` — markdown 报告，每条件推荐最佳方法
- [x] 用自己的 IMG_0701/0705/0708 照片验证
- [x] Code review：修复误导 print 文案 + 标注颜色 255→160（白字在白底上不可见）
- [x] 提交：3 个 commit（速练 / 交付任务 / review 修复）

**关键发现：**
- 白纸黑字下，uniform + side_light 全部方法都有效（纸≥150 > 127，字=0），6 种结果一致性 95.8%/4.2%
- very_dark 场景纸≈89，global(127) 全黑/全白报废，Otsu 找回 4.2% 文字——固定阈值在低光下不可靠
- BINARY_INV 得到白字黑底，4.2% 恰好是文字真实占比，可直接送 findContours

**复盘三问：**
1. BINARY vs BINARY_INV 的核心区别？BINARY = 比 T 亮→白/暗→黑，INV = 比 T 亮→黑/暗→白。选哪个取决于"要找的东西比背景亮还是暗"。
2. 自适应阈值为什么在速练 1 表现好但在交付任务里翻车？速练 1 字是白色(255)、背景是灰色(180+)——文字比背景亮，自适应局部均值法天然适合找"亮的前景"。交付任务改白纸黑字后，背景白(255)远亮于文字黑(0)，所有方法都已完美分割，自适应的优势不显现。
3. 今天学到了什么跟最终项目相关的？最终项目的视频帧预处理中，二值化这一刀怎么切取决于场景的光照条件。自适应阈值是默认首选，但如果光照稳定（如俯拍桌面），全局/Otsu 更快更稳。

### Day 14 (2026-06-24) — 阶段测试 2（模块 4-6 综合）

**完成事项：**
- [x] 阶段测试 2：8.7/10 (87%)
- [x] 选择题 4/4 全对
- [x] 代码补全 2/2 全对
- [x] 实战题 Q7 部分正确（2/3）

**错题：**
- Q7 [K] — edge_preserve 场景推荐了 Canny，正解是双边滤波。Canny 是边缘检测器（输出二值图），双边滤波才是去噪同时保留边缘的滤波器。Canny 找边缘，双边滤波保护边缘——输入输出和目标都不同。

**Week 2 薄弱点回顾：**
- 模块 4 测验 50%（仿射包含平移 [K] + 画布裁切 [C] + aliasing 术语 [E]）——已在 Day 10 费曼推导中补强
- 模块 5 测验 83%（三点共线 [K] + sigma 术语 [E]）——已改善
- 模块 6 测验 83%（Sobel 方向映射 [K] + 看错题 [E]）——Sobel X/Y 方向已纠正

**复盘三问：**
1. Canny vs 双边滤波的区别是什么？Canny 输出边缘二值图（黑白线），双边滤波输出平滑去噪图（保留原图内容）。前者找边缘，后者保护边缘。
2. Q1-Q6 为什么全对？INTER_AREA 缩小/插值选择、仿射=透视特例/三点不共线、Sobel X=垂直边缘、Canny 滞后阈值、warpPerspective/medianBlur/GaussianBlur API——Day 8-13 反复练习已扎实。
3. Week 2 最大的收获是什么？从单一操作（旋转、滤波）到组合 Pipeline（中值->高斯->Canny->透视），学会了把独立函数串成端到端流程。这才是最终项目所需要的思维方式。

### Day 13 (2026-06-24) — 组合 Pipeline：滤波->边缘->透视

**完成事项：**
- [x] 合成含噪倾斜文档测试图（make_test_document 复用）
- [x] 中值滤波去椒盐噪声（medianBlur ksize=5）
- [x] 高斯模糊进一步平滑（GaussianBlur ksize=5）
- [x] Canny 边缘检测提取文档四边
- [x] findContours 找最大四边形轮廓
- [x] approxPolyDP 逼近四角
- [x] getPerspectiveTransform + warpPerspective 透视矫正
- [x] 输出灰度图 + 边缘图并排对比

**复盘三问：**
1. Pipeline 5 个步骤的中间结果是什么？中值->椒盐噪声消失但保留边缘模糊；高斯->进一步平滑减少假边缘；Canny->白线边缘清晰但可能有噪点；最大四边形->书的四角坐标；透视矫正->正矩形俯视视角。
2. 哪个步骤最容易失败？approxPolyDP 的 epsilon 参数——太大将四边形逼近成三角形，太小保留过多噪声顶点。0.02x周长是经验起点。
3. 这条 Pipeline 和最终项目的关系？架构完全一致：逐帧处理->ROI分析->结构化输出。Day 13 Pipeline 是最终项目在单帧文档场景下的缩影。

### Day 12 (2026-06-23) — 边缘检测实验室

**完成事项：**
- [x] sobel_edges() — CV_64F -> magnitude -> convertScaleAbs 三向梯度
- [x] laplacian_edges() — 刻意不模糊展示裸 Laplacian 的噪点敏感性
- [x] canny_edges() — GaussianBlur -> Canny 标准管线
- [x] auto_canny() — medianx0.66 / medianx1.33 自动阈值
- [x] run_edge_benchmark() — 10 检测器计时 + Canny(50,150) 复用
- [x] build_edge_report() — GridSpec 2x5 对比图
- [x] 模块 6 测验：5/6 (83%)

**错题：**
- Q1 [E] -> 选 B（一定是边缘），正解 C（弱边缘看滞后连接）。看错题——梯度 120 卡在 50 和 150 之间，是 weak edge。
- Q2 [K] -> Sobel X 检测垂直边缘，Sobel Y 检测水平边缘。把算子方向和边缘方向搞反了。

**复盘三问：**
1. Sobel X 为什么检测垂直边缘？Sobel X 算的是"右边减左边"（沿 X 轴的变化率），亮度沿 X 方向跳变的像素排列恰好是竖线。
2. Canny 滞后连接为什么不用一刀切？双阈值把像素分成三档——强/弱/非——弱边缘必须挂在强边缘上才能活，既干净又连续。
3. Laplacian 真正的用武之地？图像锐化——原图减去 Laplacian 拉大边缘对比度，不是边缘检测。

### Day 11 (2026-06-22) — 滤波器对比实验室

**完成事项：**
- [x] apply_filter() — dispatch dict 分发 Gaussian/Median/Bilateral/Mean 四种滤波器
- [x] count_canny_edges() — BGR->GRAY->Canny->countNonZero 边缘保留度量
- [x] run_filter_benchmark() — perf_counter 计时，每次从 original 出发避免串扰
- [x] build_filter_report() — GridSpec 3x3 报告图 + 底部 bar chart
- [x] 模块 5 测验：5/6 (83%)

**关键发现：**
- 双边滤波边缘保留率 98.2% 但耗时 47.1ms——不可分离卷积 + 每像素 exp 运算是代价
- 中值滤波对椒盐噪声的清洗效果明显优于同等耗时的高斯 5x5
- 高斯 15x15 边缘仅保留 5.3%，核变大边缘损耗指数级增长

### Day 10 (2026-06-22) — 透视变换 4 点矫正

**完成事项：**
- [x] 费曼学习法拆解透视变换、高斯/中值/双边滤波三个概念
- [x] plt.ginput 交互选 4 个文档角 + 中键撤销
- [x] getPerspectiveTransform + warpPerspective 矫正
- [x] 透视变换测验：4/5 (80%)

**错题：**
- Q2 [K] -> 4 个源点的几何约束选了"必须顺时针"而非"三点不共线"。顺时针只是习惯，三点共线才会导致方程线性相关报错。

### Day 9 (2026-06-18) — 仿射变换 + 图像金字塔

**完成事项：**
- [x] safe_rotate() — warpAffine 旋转 + 自动扩画布
- [x] build_pyramid() + build_pyramid_collage()
- [x] build_rotation_collage() — 0-330 每 30 旋转
- [x] 模块 4 测验：2.5/5 (50%)

**错题：**
- Q3 [K] -> 仿射变换包含平移变换（不是互不包含）。仿射 = 线性变换 + 平移。
- Q4 [C] -> 旋转 45 画布不变一定会裁内容。对角线 = 边长x/2 > 边长。
- Q5 [E] -> 方向对但缺关键术语"混叠效应（aliasing）"。

### Day 8 (2026-06-18) — 图像缩放与 letterbox

**完成事项：**
- [x] letterbox_resize() — 保持纵横比缩放 + 黑色居中填充
- [x] process_batch() — 批量处理 + 统计收集
- [x] 8/8 输出严格 224x224，无拉伸变形

### Day 7 (2026-06-17) — 阶段测试 1（模块 1-3 综合）

**完成事项：**
- [x] 阶段测试 1：6/7 (86%)
- [x] 选择题 4/4 全对
- [x] 代码补全 2/2 全对
- [ ] 实战题 Q7 未通过

**错题：**
- Q7 [E]+[A] — 语法错误 + 未用题目给定阈值，用 Day 4 bimodal 算法替代

### Day 6 (2026-06-17) — HSV 色彩空间与 inRange 颜色过滤

**完成事项：**
- [x] hsv_presets() — 5 种颜色的 HSV 预设字典
- [x] apply_color_filter() + extract_objects() + count_objects()
- [x] build_filter_report() — GridSpec 5x3 报告图

### Day 5 (2026-06-16) — 图像混合与差异检测

**完成事项：**
- [x] blend_images() — cv2.addWeighted alpha 混合
- [x] difference_map() — cv2.absdiff + threshold(30) 过滤噪声
- [x] Day 5 测验：5/5 (100%)

### Day 4 (2026-06-15) — 直方图计算与均衡化

**完成事项：**
- [x] cv2.calcHist vs np.histogram 一致性验证
- [x] assess_exposure() — 百分位数决策链
- [x] equalize_color() — HSV 空间只调 V 通道
- [x] Day 4 测验：4.5/5 (90%)

### Day 3 (2026-06-11) — ROI 切片与九宫格

**完成事项：**
- [x] 中心 500x500 ROI + 共享内存验证
- [x] 3x3 九宫格 + permutation + copyMakeBorder + 拼接
- [x] Day 3 测验：4/5 (80%)

### Day 2 (2026-06-11) — 色彩空间与通道操作

**完成事项：**
- [x] BGR/RGB/HSV/Gray 转换 + split + applyColorMap
- [x] Day 2 测验：5/5 (100%)

### Day 1 (2026-06-10) — 像素矩阵探索

**完成事项：**
- [x] MP4 提取帧 + shape/dtype/size + top-5 最亮像素 + 水印
- [x] Day 1 测验：3/5 (60%)

### Day 0 (2026-06-09) — 环境准备

**完成事项：**
- [x] 阅读 CLAUDE.md、domain_map.md、30_day_roadmap.md
- [x] 验证环境：OpenCV 4.13.0 可正常导入
- [x] 选定最终项目：具身视觉数据管道（MP4 -> JSON/NPZ 结构化观察数据）

---

## 错题本

| 日期 | 模块 | 题号 | 我的答案 | 正确答案 | 错误类型 | 正确解法 | 已重做 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 06-10 | 模块 1 | Q2 | COLOR_GRAY2BGR | COLOR_BGR2RGB | [K] | BGR2RGB 交换通道，GRAY2BGR 是升维 | - |
| 06-10 | 模块 1 | Q4 | A (3840,2160,3) | C (2160,3840) | [K] | shape 永远是 (高, 宽) | - |
| 06-11 | 模块 2 | Q3 | "没有丢弃像素" | 720x1280 | [E] | 没读完题 | - |
| 06-15 | 模块 2 | Q5 | "只能调 V 通道" | BGR 分通道均衡化破坏颜色比例 | [E] | 理由不够精准 | - |
| 06-17 | 测试 1 | Q7 | Day 4 bimodal 算法 | p50<60 or p95<150 | [E]+[A] | 语法错误 + 未用题目阈值 | - |
| 06-18 | 模块 4 | Q3 | 仿射和透视互换 | 仿射包含平移 | [K] | 仿射 = 线性 + 平移 | - |
| 06-18 | 模块 4 | Q4 | 旋转45不会一定裁 | 对（画布不变一定裁） | [C] | 对角线 > 边长 | - |
| 06-18 | 模块 4 | Q5 | 细节混在一起 | 混叠效应(aliasing) | [E] | 高频分量直接下采样产生锯齿 | - |
| 06-24 | 测试 2 | Q7 | Canny（边缘检测器）| 双边滤波（边缘保留去噪）| [K] | Canny 找边缘，双边滤波保护边缘 | - |
| 07-17 | 模块 9 | Q2 | "threshold 越大直线越直" | threshold 越大直线越少 | [K] | Hough threshold 是最少投票数，越大=越严格，通过的直线越少 | - |

---

## 薄弱点追踪

| 发现日期 | 概念 | 出现模块 | 错误次数 | 计划补救 | 状态 |
|:---|:---|:---|:---|:---|:---|
| 06-10 | shape 维度顺序 (高,宽) vs (宽,高) | 模块 1 | 1 | Day 2 加深 BGR 索引练习 | 已改善 |
| 06-11 | 非整除图片的边角处理 | 模块 2 | 0 | Day 4 练习中测试非整除尺寸 | 已改善 |
| 06-15 | 曝光判断：均值 vs 百分位数选择 | 模块 2 | 0 | 亮度自适应场景中强化 | 待观察 |
| 06-15 | 彩色均衡化：HSV-V 而非 BGR 分通道 | 模块 2 | 1 | Day 6 HSV 模块覆盖 | 已改善 |
| 06-17 | 做题时擅自优化题目给定规则 | 测试 1 | 1 | 下次测试前提醒 | 待观察 |
| 06-18 | 仿射变换数学直觉（2x3 矩阵几何意义）| 模块 4 | 1 | Day 10 费曼推导补强 | 已改善 |
| 07-17 | Hough threshold 含义（越大越"少"而非"直"）| 模块 9 | 1 | Day 20 Pipeline 综合练习中强化 | 待观察 |
| 06-24 | Canny vs 双边滤波作用混淆 | 测试 2 | 1 | Week 3 轮廓分析中区分边缘检测和滤波 | 待观察 |

---

## 阶段测试成绩

| 测试 | 日期 | 选择题 | 代码补全 | 实战题 | 总分 | 薄弱环节 |
|:---|:---|:---|:---|:---|:---|:---|
| 测试 1 (Day 7) | 06-17 | 4/4 | 2/2 | 0/1 | 6/7 (86%) | Q7 [E]+[A] |
| 测试 2 (Day 14) | 06-24 | 4/4 | 2/2 | 0.67/1 | 8.7/10 (87%) | Q7 [K] Canny vs 双边滤波 |

### 模块测验成绩

| 模块 | 日期 | 得分 | 薄弱环节 |
|:---|:---|:---|:---|
| 模块 1 | Day 1-2 | 80% (8/10) | BGR 通道索引 [K], shape [K] |
| 模块 2 | Day 3-4 | 85% (8.5/10) | 非整除边角处理 [E], 均衡化理由 [E] |
| 模块 3 | Day 5-6 | 100% (5/5) | — |
| 模块 4 | Day 8-9 | 50% (2.5/5) | 仿射 = 线性+平移 [K], 画布不变必裁 [C], aliasing [E] |
| 模块 5 | Day 10-11 | 83% (5/6) | 三点共线 [K], sigma 术语 [E] |
| 模块 6 | Day 12 | 83% (5/6) | Sobel X/Y 方向映射 [K], 看错题 [E] |
| 模块 7 | Day 15-16 | 90% (4.5/5) | 二值化在 Pipeline 中的位置 [A] |
| 模块 8 | Day 17-18 | 100% (5/5) | — |
| 模块 9 | Day 19 | 75% (6/8) | Hough threshold 含义混淆 [K] |

---

## 最终项目 — 具身视觉数据管道

### 项目描述
输入一段桌面操作视频（MP4），程序自动完成：帧提取 -> 多 ROI 分窗 -> 每 ROI 独立分析 -> 帧间物体追踪 -> 结构化数据打包（JSON + NPZ）。

### 项目进度

| 日期 | 完成内容 | 遇到的问题 | 解决方案 |
|:---|:---|:---|:---|
| Day 24 | 视频 I/O 基础 | - | - |
| Day 25 | 运动检测追踪 | - | - |
| Day 26 | 多 ROI 分析 | - | - |
| Day 27 | 数据管道 v0.1 | - | - |
| Day 28 | 阶段测试 4 + 框架搭建 | - | - |
| Day 29 | 全流程整合 | - | - |
| Day 30 | 完善 + 演示 | - | - |

### 项目验收

- [ ] `python experiments/day_30_demo.py data/raw/my_video.mp4` 一行命令跑通
- [ ] 自动从 MP4 中提取帧（可配置帧间隔）
- [ ] 至少 3 个 ROI 窗口各自完成特征提取
- [ ] 帧间物体追踪轨迹可视化
- [ ] 输出包含：JSON + NPZ + 可视化视频
- [ ] 录制了完整的演示视频

---

## 完成统计

```
Week 1 图像基石:     7/7 天  ✓
Week 2 图像变换:     7/7 天  ✓
Week 3 图像分析:     6/7 天
Week 4 进阶+项目:    0/9 天  (含 Day 29-30 项目冲刺)
----------------------------------------------
总进度:             20/30 天
```

---

## 下一步计划

- [x] Day 14: 阶段测试 2 — 87% (8.7/10)
- [x] Day 15: 模块 7 概念 A — 二值化（全局/Otsu/自适应）+ BINARY vs BINARY_INV
- [x] Day 16: 模块 7 概念 B+C — 腐蚀/膨胀/开运算/闭运算 + 结构元素对比
- [x] **模块 7 测验** — 90% (4.5/5) Q5 二值化放在透视前 [A]
- [x] Day 17: 模块 8 概念 A — 轮廓提取（findContours + drawContours + 层级）
- [x] Day 18: 模块 8 概念 B+C — 轮廓几何属性 + approxPolyDP 形状分类 + 模块 8 测验
- [x] Day 19: 模块 9 — Hough 变换（直线检测 + 圆检测）
- [x] Day 20: 模块 9 概念 C — Pipeline 设计 + 综合练习（成功验证 3 张真实图 + 4 个视频帧）
