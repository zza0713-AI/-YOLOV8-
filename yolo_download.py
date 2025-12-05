from ultralytics import YOLO

# 加载模型（首次运行会自动下载 yolov8l.pt，后续直接加载本地文件）
model = YOLO("yolov8m.pt")  # 若已下载，直接读取本地权重

# 运行检测（source 替换为你的图片/视频路径）
results = model.predict(
    source="C:\\Users\\ang32\\Desktop\\60588970bd8d8e3f07a5434a4f86cf98.png",  # 本地图片路径（如无则先放一张图片到同级目录）
    save=True,            # 保存检测结果
    conf=0.5,             # 置信度阈值（只显示置信度≥0.5的目标）
    iou=0.45              # 重叠目标过滤阈值
)

# 打印检测结果（可选）
print(results[0].boxes)  # 输出目标框坐标、类别、置信度
