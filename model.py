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



class Detector(nn.Module):
    def __init__(self, num_keypoints=0) -> None:
        super(Detector, self).__init__()

        self.feature_extractor = nn.Sequential(
            ResNetBlock(1),
            ResNetBlock(16)
        )
        self.feature_extractor_ = FeatureExtractor(1, 5)
        self.scale_variant = ScaleInvariant()

    def forward(self, x):
        features = self.feature_extractor_(x)
        display_tensor(features, 'visualization/features', 'features')
        score_map = self.scale_variant(features)

        return score_map
    
if __name__ == '__main__':
    scale_network = FeatureExtractor(1, 5).cuda()
    input_size = (1, 512, 512)
    summary(scale_network, input_size=input_size)
