"""Poster Visual Feature Extractor using ResNet50"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os
from typing import List, Dict
from tqdm import tqdm
import pickle
class PosterFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=128, device='cpu', freeze_initial=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.current_epoch = 0
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if freeze_initial:
            self._freeze_resnet()
        self.to(self.device)
    def _freeze_resnet(self):
        for param in self.resnet50.parameters():
            param.requires_grad = False
    def _unfreeze_resnet(self):
        for param in self.resnet50.parameters():
            param.requires_grad = True
    def set_epoch(self, epoch):
        if epoch >= 5 and all(not p.requires_grad for p in self.resnet50.parameters()):
            self._unfreeze_resnet()
    def forward(self, images):
        with torch.no_grad() if not any(p.requires_grad for p in self.resnet50.parameters()) else torch.enable_grad():
            features = self.resnet50(images)
        features = features.view(features.size(0), -1)
        return self.projection(features)
    def extract_features_from_paths(self, image_paths, batch_size=32):
        all_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size)):
                batch_images = []
                for path in image_paths[i:i+batch_size]:
                    try:
                        image = Image.open(path).convert('RGB') if os.path.exists(path) else Image.new('RGB', (224, 224))
                        batch_images.append(self.transform(image))
                    except:
                        batch_images.append(torch.zeros(3, 224, 224))
                if batch_images:
                    batch = torch.stack(batch_images).to(self.device)
                    all_features.append(self(batch).cpu().numpy())
        return np.vstack(all_features).astype(np.float32) if all_features else np.zeros((len(image_paths), self.embed_dim))
