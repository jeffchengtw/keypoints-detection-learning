import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import Detector, STNSampler, DescriptorNetwork
from viz import draw_keypoints, extract_keypoints, display_patches

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper
num_keypoints = 1000
num_epochs = 100
predict_interval = 10

train_dataset = MyDataset(r'sample')
train_loader = DataLoader(train_dataset, batch_size=1)

detector_i = Detector(num_keypoints).to(device)
detector_j = Detector(num_keypoints).to(device)
sampler = STNSampler().to(device)
descriptor_i = DescriptorNetwork().to(device)
descriptor_j = DescriptorNetwork().to(device)

criterion_img = nn.MSELoss()
criterion_D = nn.MSELoss()

optimizer_detector = torch.optim.Adam(detector_i.parameters(), lr=0.001)
optimizer_descriptor = torch.optim.Adam(descriptor_i.parameters(), lr=0.001)


best_loss = 999999999

for epoch in range(num_epochs):
    detector_i.train()
    detector_j.train()
    running_loss = 0.
    for data in train_loader:
        data = data.to(device)
        optimizer_detector.zero_grad()
        optimizer_descriptor.zero_grad()
        
        # model i forward
        score_map_i, orientation_map_i = detector_i(data)
        kps_i = extract_keypoints(score_map_i, num_keypoints)
        draw_keypoints(score_map_i, kps_i, r'visualization\features', f'kps_i_{epoch}')
        
        # model j forward
        rotated_data = torch.rot90(data, 1, dims=(2, 3))
        score_map_j, orientation_map_j = detector_j(rotated_data)
        kps_j = extract_keypoints(score_map_j, num_keypoints)
        draw_keypoints(score_map_j, kps_j, r'visualization\features', f'kps_j_{epoch}')
        reversed_score_map_j = torch.rot90(score_map_j, -1, dims=(2, 3))

        # compute image level loss
        loss_img = criterion_img(score_map_i, reversed_score_map_j)

        # STN sampler
        rotated_score_map_i = torch.rot90(score_map_i, 1, dims=(2, 3))
        rotated_orientation_map_i = torch.rot90(orientation_map_i, 1, dims=(2, 3))
        rotated_kps_i = extract_keypoints(rotated_score_map_i, num_keypoints)
        p_i = torch.cat([data, score_map_i, orientation_map_i], dim=1)
        p_j = torch.cat([rotated_data, rotated_score_map_i, rotated_orientation_map_i], dim=1)

        patches_i = sampler(p_i, kps_i)
        patches_j = sampler(p_j, rotated_kps_i)

        # descriptor
        D_i = descriptor_i(patches_i)
        D_j = descriptor_j(patches_j)
        
        # compute descroptor loss
        loss_D = criterion_D(D_i, D_j)

        # compute detector loss
        L_det = loss_D + loss_img

        # backward
        L_det.backward()
        # assign to detector j 
        detector_j.load_state_dict(detector_i.state_dict())
        descriptor_j.load_state_dict(descriptor_i.state_dict())
        # update detector i 
        optimizer_detector.step()
        optimizer_descriptor.step()

        running_loss += L_det.item()
    epoch_loss = running_loss / num_epochs
    print(f'epoch : {epoch} current loss : {epoch_loss}')
    
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        print(f'loss improved in epoch{epoch}')
        dst_path = f'ckpt\epoch{epoch}.pt'
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        torch.save({
            'detector_state_dict': detector_i.state_dict(),
            'descriptor_state_dict': descriptor_i.state_dict(),
            'epoch' : epoch,
            }, dst_path)


        



