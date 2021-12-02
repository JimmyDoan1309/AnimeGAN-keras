import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def edge_smooth(image, target_size=None, color_mode='RGB'):
    if target_size is not None:
        image = cv2.resize(image, target_size)
    
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    
    if color_mode == 'RGB':
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif color_mode == 'BGR':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pad = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='reflect')
    
    edges = cv2.Canny(gray, 100, 200)
    dilation = cv2.dilate(edges, kernel)
    
    gauss_img = np.copy(image)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
            np.multiply(pad[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
            np.multiply(pad[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
            np.multiply(pad[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))
        
    return gauss_img

def edge_smooth_files(source_path, saved_path, target_size=None, workers=4, override=False):
    assert os.path.exists(source_path), f'Path {source_path} does not exist'
    
    files = os.listdir(source_path)
    assert len(files) > 0, f'No file in {source_path}'
    
    os.makedirs(saved_path, exist_ok=True)
    
    def _task(file):
        image = cv2.imread(os.path.join(source_path, file))
        smooth = edge_smooth(image, target_size, color_mode='BGR')
        cv2.imwrite(os.path.join(saved_path, file), smooth)
    
    with ThreadPoolExecutor(workers) as executor:
        with tqdm(total=len(files)) as pbar:
            futures = []
            for file in files:
                if not override and os.path.exists(os.path.join(saved_path, file)):
                    pbar.update(1)
                    continue
                future = executor.submit(_task, file)
                future.add_done_callback(lambda _: pbar.update(1))
                futures.append(future)
            for future in futures:
                future.result()
    