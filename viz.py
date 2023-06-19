import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def extract_keypoints(scale_space_score_map, num_keypoints=50):
        batch_size, _, height, width = scale_space_score_map.shape

        score_map = scale_space_score_map.squeeze(dim=1)

        _, indices = torch.topk(score_map.view(batch_size, -1), k=num_keypoints, dim=1)

        keypoints = []
        for i in range(batch_size):
            indices_i = indices[i]
            row_indices = indices_i // width
            col_indices = indices_i % width
            keypoints_i = torch.stack([col_indices.float(), row_indices.float()], dim=1)
            keypoints.append(keypoints_i)

        keypoints = torch.stack(keypoints, dim=0).float()
        return keypoints

def draw_keypoints(score_maps, keypoints, dst_path, filename):
    score = score_maps.clone().cpu()
    image = tensor_to_image(score)
    batch_size, num_keypoints, _,  = keypoints.shape

    for i in range(batch_size):
        image = image[i].squeeze()  # 去除维度为1的维度
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for j in range(num_keypoints):
            prediction = keypoints[i, j]
            x = int(prediction[0].item())
            y = int(prediction[1].item())

            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 在图像上绘制关键点

        # 確保目錄存在，如果不存在則創建
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path + f'\{filename}_{i}.png', image)

def display_tensor(input_tensor: torch.Tensor, path: str, filename):
    """Save tensor to image

    Args:
        input_tensor (torch.Tensor): Input tensor (b, c, h, w)
        path (str): Destination path
    """
    tensor = input_tensor.clone()
    tensor = tensor.cpu()
    
    # 將其轉換為 OpenCV 所使用的圖片格式 (h, w, c)
    for i in range(tensor.shape[1]):
        image = tensor_to_image(tensor[:, i, :, :])
        
        # 儲存圖片到指定路徑，每張圖片以索引命名
        save_image(image, os.path.join(path, f"{filename}_{i}.png"))
    
def tensor_to_image(tensor: torch.Tensor):
    """Convert tensor to grayscale image

    Args:
        tensor (torch.Tensor): Input tensor (b, h, w)

    Returns:
        np.ndarray: Converted image
    """
    
    # 轉換為 numpy array
    image = tensor.detach().numpy()
    
    # 將數值範圍轉換為 0-255 並轉換為 8-bit 圖片
    tensor_min = image.min()
    tensor_max = image.max()
    image = (image - tensor_min) / (tensor_max - tensor_min) * 255
    image = image.astype(np.uint8)
    return image

def save_image(image: np.ndarray, path: str):
    """Save image to disk

    Args:
        image (np.ndarray): Input image
        path (str): Destination path
    """
    # 確保目錄存在，如果不存在則創建
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 儲存圖片到指定路徑
    image = image[0, :, :]
    cv2.imwrite(path, image)

def display_patches(input_patch: torch.Tensor, path: str):
    b, n, h, w, c = input_patch.size()

    for i in range(n):
        patch = input_patch[:, i].clone()  # 克隆第 i 個 patch
    
        gray = patch[:, 0].unsqueeze(1)  # 取得第 j 個通道
        display_tensor(gray, path, f'pathes{i}')  # 儲存通道圖像