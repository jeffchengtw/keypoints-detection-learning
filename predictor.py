import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from model import Detector, STNSampler, DescriptorNetwork
from dataset import MyDataset
from viz import draw_keypoints, extract_keypoints, display_patches


def feature_matching(img1, img2, des1, des2, kp1, kp2):
    # 讀取圖像
    image1 = img1
    image2 = img2

    # 初始化SIFT
    sift = cv2.SIFT_create()
    keypoints1_ = [cv2.KeyPoint(x, y, 1) for x, y in kp1]
    keypoints2_ = [cv2.KeyPoint(x, y, 1) for x, y in kp2]
    # 檢測特徵點和計算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 創建BFMatcher對象
    bf = cv2.BFMatcher()

    # 進行特徵匹配
    matches = bf.match(des1, des2)

    # 根據距離排序匹配結果
    matches = sorted(matches, key=lambda x: x.distance)

    # 繪製前10個匹配結果
    result = cv2.drawMatches(image1, keypoints1_, image2, keypoints2_, matches[:2000], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 返回匹配結果圖像
    return result

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_dir = 'ckpt/epoch3.pt'
num_keypoints = 5000

# create dataloader
test_dataset = MyDataset(r'test')
test_loader = DataLoader(test_dataset, batch_size=1)

# create model
detector = Detector(num_keypoints).to(device)
descriptor = DescriptorNetwork().to(device)

# load checkpoint
checkpoint = torch.load(ckpt_dir)
detector.load_state_dict(checkpoint['detector_state_dict'])
descriptor.load_state_dict(checkpoint['descriptor_state_dict'])
sampler = STNSampler().to(device)
# inference
img_list = []
des = []
kps = []
detector.eval()
descriptor.eval()
for data_dict in test_loader:
    data = data_dict['gray_tensor'].to(device)
    # model i forward
    score_map_i, orientation_map_i = detector(data)
    kps_i = extract_keypoints(score_map_i, num_keypoints)
    draw_keypoints(score_map_i, kps_i, r'visualization/pred', data_dict['filename'][0] + '_DLresult', data_dict['bgr_arr'])

    # des
    p_i = torch.cat([data, score_map_i, orientation_map_i], dim=1)
    patches_i = sampler(p_i, kps_i)
    D_i = descriptor(patches_i)
    # opencv result
    bgr_tensor = data_dict['bgr_arr']
    bgr_array = np.array(bgr_tensor)
    bgr_array = bgr_array[0]

    kps_np = kps_i.detach().cpu().numpy()
    kps_reshaped = np.reshape(kps_np, (kps_np.shape[1], kps_np.shape[2]))
    des_np = np.array(D_i.detach().cpu(), dtype=np.float32)
    img_list.append(bgr_array)
    des.append(des_np)
    kps.append(kps_reshaped)

img1 = img_list[0]
img2 = img_list[1]
des1 = des[0]
des2 = des[1]
kp1, kp2 = kps[0], kps[1]
res = feature_matching(img1, img2, des1, des2, kp1, kp2)
cv2.imwrite(r'visualization/pred/' + data_dict['filename'][0] + '_DLresult.png', res)