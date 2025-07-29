import matplotlib.pyplot as plt
from dataset import CelebA

train = CelebA("train")
figure = plt.figure(figsize=(8, 8))
for i in range(9):
    img, identity = train[i]
    figure.add_subplot(3, 3, i + 1)
    plt.title(f"Identity: {identity}")
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0).numpy())
plt.show()
