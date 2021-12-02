import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

def get_image(url, as_numpy=False):
    if url.startswith('http'):
        resp = requests.get(url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert('RGB')
    else:
        img = Image.open(url).convert('RGB')
    
    if as_numpy:
        return np.array(img)
    return img

def plot_image(image, gray=False, figsize=(7,7)):
    if np.max(image) > 1:
        image = image.astype(np.uint8)
    
    plt.figure(figsize=figsize)
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    
def normalize(image):
    image = image / 127.5
    image = image - 1.0
    return image

def denormalize(image, as_float=True):
    image = ((image+1)*127.5)/255.0
    image = np.clip(image, 0, 1)
    if not as_float:
        image = image*255
        image = image.astype(np.uint8)
    return image
