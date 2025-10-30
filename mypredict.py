from ultralytics import YOLO

model=YOLO("yolov8n.pt")

model.predict(
    source=r"C:\Users\GSH\Desktop\D9BDCDB10C55902F214FCA2D094AAE1C.jpg.gif",
    save=True,
    show=False,
)