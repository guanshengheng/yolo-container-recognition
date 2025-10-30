import cv2
from ultralytics import YOLO

# 加载模型
# 注意：图片中的 "yolo11n.pt" 可能需要替换为您实际拥有的模型文件，
# 例如 "yolov8n.pt"，YOLO会自动下载标准模型。
model = YOLO("yolov8n.pt")
# 或者使用您图片中的路径: model = YOLO(r"yolo11n.pt")

# 从摄像头 (source=0) 进行流式推理
results = model(
    source=0,
    stream=True,
)

# 循环处理每一帧
for result in results:
    # 绘制检测框
    plotted = result.plot()

    # 显示图像 (修正了图片中 "winname:" 的语法)
    cv2.imshow("YOLO Inference", plotted)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 关闭所有窗口
cv2.destroyAllWindows()