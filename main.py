import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import Detector
from viz import draw_keypoints, extract_keypoints

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper
num_keypoints = 5000
num_epochs = 100
predict_interval = 10

train_dataset = MyDataset(r'data\mountain')
train_loader = DataLoader(train_dataset, batch_size=1)

model_i = Detector(num_keypoints).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_i.parameters(), lr=0.001)

initial_params = {}


for epoch in range(num_epochs):
    model_i.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        score_map_i = model_i(data)
        kps = extract_keypoints(score_map_i, num_keypoints)
        draw_keypoints(score_map_i, kps, r'D:\NCTU\CV\Final\visualization\features', 'kps')
        # loss = criterion(pred_x, train_y)

        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
              1, num_epochs, loss.item()))
