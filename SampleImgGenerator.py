import numpy as np
from PIL import Image

img_size = {"height": 60, "width": 80, "depth": 3}

if __name__ == "__main__":
    for i in range(16):
        img_array = np.random.randint(0, 256, (img_size["height"], img_size["width"], img_size["depth"]), dtype=np.uint8)
        Image.fromarray(img_array).save("./TestImg/" + str(i) + ".jpg")
        print(img_array.shape, img_array.dtype, np.max(img_array), np.min(img_array), "./Img/" + str(i) + ".jpg")
