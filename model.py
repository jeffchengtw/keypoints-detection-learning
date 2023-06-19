import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from viz import display_tensor


class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.leaky_relu(out)

        return out

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.leaky_relu(out)

        return out

class ScaleInvariantDetector(nn.Module):
    def __init__(self, num_scales=5, scale_factor=2 ** 0.5):
        super(ScaleInvariantDetector, self).__init__()
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.convolution_layers = nn.ModuleList(
            [nn.Conv2d(5, 1, kernel_size=5, padding=2) for _ in range(num_scales)])

    def forward(self, feature_map):
        batch_size, num_channels, height, width = feature_map.shape
        device = feature_map.device

        response_maps = []
        scales = [self.scale_factor ** i for i in range(self.num_scales)]

        for scale_idx, scale in enumerate(scales):
            resized_feature_map = nn.functional.interpolate(
                feature_map, scale_factor=1/scale, mode='bilinear', align_corners=False)
            response_map = self.convolution_layers[scale_idx](
                resized_feature_map)
            response_map = nn.functional.interpolate(response_map, size=(
                height, width), mode='bilinear', align_corners=False)
            response_maps.append(response_map)
            display_tensor(resized_feature_map, 'visualization/features', f'resize_features{scale_idx}')

        # shape: (batch_size, num_scales, height, width)
        final_score_map = torch.cat(response_maps, dim=1)
        # merge the h¯n into final scale-space score map
        scale_space_score_map = torch.sum(final_score_map, dim=1, keepdim=True)

        return scale_space_score_map

class ScaleInvariant(nn.Module):
    def __init__(self, num_scales=5, scale_factor=2 ** 0.5):
        super(ScaleInvariant, self).__init__()
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.convolution_layers = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=5, padding=2) for _ in range(num_scales)])
    
    def convolutional_softmax(self, score_map):
        b, c, h, w = score_map.shape
        
        # Define the window size
        window_size = 15
        
        # Calculate padding size
        padding = (window_size - 1) // 2
        
        # Reshape score_map to have a single channel dimension
        score_map = score_map.view(b, -1, h, w)
        
        # Apply convolutional softmax over 15x15 windows
        softmax_map = F.softmax(score_map, dim=2)
        softmax_map = F.softmax(softmax_map, dim=3)
        
        return softmax_map
    
    def forward(self, feature_maps):
        batch_size, num_channels, height, width = feature_maps.shape

        response_maps = []
        response_maps_2 = []
        scales = [self.scale_factor ** i for i in range(self.num_scales)]
        for scale_idx, scale in enumerate(scales):
            scale_factor = 1 / scale
            feature_map = torch.split(feature_maps, 1, dim=1)[scale_idx]
            resized_feature_map = nn.functional.interpolate(feature_map, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            display_tensor(resized_feature_map, 'visualization/features', f'resize_features{scale_idx}')
            response_map = self.convolution_layers[scale_idx](resized_feature_map)
            response_map = self.convolutional_softmax(response_map)
            response_map = nn.functional.interpolate(response_map, size=(height, width), mode='bilinear', align_corners=False)
            response_maps.append(response_map)
        
        # shape: (batch_size, num_scales, height, width)
        final_score_map = torch.cat(response_maps, dim=1)
        # merge the h¯n into final scale-space score map
        scale_space_score_map = torch.sum(final_score_map, dim=1, keepdim=True)
        display_tensor(scale_space_score_map, 'visualization/features', f'score_map')

        return scale_space_score_map

class OrientationConv(nn.Module):
    def __init__(self) -> None:
        super(OrientationConv, self).__init__()

        self.orientation_conv = nn.Conv2d(5, 2, kernel_size=5, padding=2)

    def forward(self, feature_maps):
        orientation = self.orientation_conv(feature_maps)
        sin = torch.sin(orientation)
        cos = torch.cos(orientation)
        theta = torch.atan2(sin, cos)
        return theta
    
class Detector(nn.Module):
    def __init__(self, num_keypoints=0) -> None:
        super(Detector, self).__init__()

        self.feature_extractor = nn.Sequential(
            ResNetBlock(1),
            ResNetBlock(16)
        )
        self.feature_extractor_ = FeatureExtractor(1, 5)
        self.scale_variant = ScaleInvariant()
        self.orientation_extractor = OrientationConv()

    def forward(self, x):
        features = self.feature_extractor_(x)
        display_tensor(features, 'visualization/features', 'features')
        score_map = self.scale_variant(features)
        orientation_map = self.orientation_extractor(features)

        return score_map

class STN(nn.Module):
    def __init__(self, input_size):
        super(STN, self).__init__()
        self.input_size = input_size

        self.localization = nn.Sequential(
            nn.Conv2d(input_size, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

    def forward(self, x, keypoints):
        # 提取關鍵點位置
        keypoints_x = keypoints[:, 0].view(-1, 1, 1)
        keypoints_y = keypoints[:, 1].view(-1, 1, 1)

        # 生成仿射矩陣
        theta = self.fc_loc(self.localization(x).view(-1, 10 * 4 * 4))
        theta = theta.view(-1, 2, 3)

        # 執行可微分的空間轉換
        grid = F.affine_grid(theta, x.size())
        cropped_image = F.grid_sample(x, grid)

        # 根據關鍵點位置裁剪圖像
        cropped_patches = cropped_image[:, :, keypoints_y, keypoints_x]

        return cropped_patches

class SimpleDesc(nn.Module):
    def __init__(self, input_channels, out_dim=256):
        super(SimpleDesc, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)  # Adjusted input size
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        descriptors = F.normalize(x, p=2, dim=1)
        return descriptors

    
if __name__ == '__main__':
    siamese_model = SimpleDesc(input_channels=1).cuda()
    input_size = (1, 32, 32)
    summary(siamese_model, input_size=input_size)
