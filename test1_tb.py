from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img=Image.open("datasets/coco8/images/train/000000000009.jpg")
print(img)
#  ToTensor使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# ...
writer.add_image("ToTensor", img_tensor)

# Normalize使用
# ↓↓↓ 添加 "Before Norm: " 标签
print(f"Before Norm (img_tensor): {img_tensor[0][0][0]}")

trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# -----------------------------------------------
# 实验：您可以把上面这行改成：
# trans_norm = transforms.Normalize([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
# -----------------------------------------------

img_norm = trans_norm(img_tensor)

# ↓↓↓ 添加 "After Norm: " 标签
print(f"After Norm (img_norm): {img_norm[0][0][0]}")

writer.add_image("Normalize", img_norm)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
print(img_resize)
writer.add_image("Resize", img_resize)
writer.close()