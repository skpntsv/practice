from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os

# 1) Загрузить MNIST тестовый набор
mnist = datasets.MNIST(root='.', train=False, download=True)

# 2) Получить первый образец
img, label = mnist[1]  # img – PIL Image в режиме 'L' (градации серого)

imgL = img.convert('L')
os.makedirs('data', exist_ok=True)
imgL.save('./sample.bmp')
print("Saved ./sample.bmp, label:", label)
