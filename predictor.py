import torch
from torch.utils.data import DataLoader
from model import Detector, DescriptorNetwork
from dataset import MyDataset
from viz import draw_keypoints, extract_keypoints, display_patches


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_dir = 'ckpt/best_model.pt'
num_keypoints = 1000

# create dataloader
test_dataset = MyDataset(r'sample')
test_loader = DataLoader(test_dataset, batch_size=1)

# create model
detector = Detector(num_keypoints).to(device)
descriptor = DescriptorNetwork().to(device)

# load checkpoint
checkpoint = torch.load(ckpt_dir)
detector.load_state_dict(checkpoint['detector_state_dict'])
descriptor.load_state_dict(checkpoint['descriptor_state_dict'])

# inference
for data_dict in test_loader:
    data = data_dict['gray_tensor'].to(device)
    # model i forward
    score_map_i, orientation_map_i = detector(data)
    kps_i = extract_keypoints(score_map_i, num_keypoints)
    draw_keypoints(score_map_i, kps_i, r'visualization/pred', data_dict['filename'], data_dict['bgr_arr'])
