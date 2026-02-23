import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

class Squash(nn.Module):
    """
    Applies the non-linear squashing function to ensure the vector length 
    is scaled between 0 and 1 without changing its orientation.
    """
    def __init__(self, epsilon=1e-8):
        super(Squash, self).__init__()
        self.epsilon = epsilon

    def forward(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + self.epsilon)

class PrimaryCapsule(nn.Module):
    """
    Transforms the fused feature vector into a set of primary capsules.
    """
    def __init__(self, in_features=1472, num_capsules=32, dim_capsule=8):
        super(PrimaryCapsule, self).__init__()
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.dense = nn.Linear(in_features, num_capsules * dim_capsule)
        self.squash = Squash()

    def forward(self, x):
        # x shape: (batch_size, in_features)
        x = self.dense(x)
        x = x.view(-1, self.num_capsules, self.dim_capsule)
        return self.squash(x)

class ClassCapsule(nn.Module):
    """
    Executes the dynamic routing algorithm to compute class probabilities.
    """
    def __init__(self, num_routes=32, in_dim=8, num_classes=2, out_dim=16, routing_iters=3):
        super(ClassCapsule, self).__init__()
        self.num_classes = num_classes
        self.routing_iters = routing_iters
        
        # Transformation matrix W
        self.W = nn.Parameter(torch.randn(1, num_routes, num_classes, out_dim, in_dim) * 0.01)
        self.squash = Squash()

    def forward(self, x):
        # x shape: (batch_size, num_routes, in_dim)
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4) # (batch_size, num_routes, 1, in_dim, 1)
        
        # Predict spatial relationships
        # u_hat shape: (batch_size, num_routes, num_classes, out_dim)
        u_hat = torch.matmul(self.W, x).squeeze(4)
        
        # Dynamic Routing
        b_ij = torch.zeros(batch_size, u_hat.size(1), self.num_classes, 1).to(x.device)
        
        for i in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j, dim=-1)
            
            if i < self.routing_iters - 1:
                agreement = (u_hat * v_j).sum(dim=-1, keepdim=True)
                b_ij = b_ij + agreement
                
        return v_j.squeeze(1) # (batch_size, num_classes, out_dim)

class EVCNet(nn.Module):
    """
    EVC-Net: Hybrid framework combining EfficientNetV2S, Vision Transformer, 
    and Capsule Networks for histopathological image classification.
    """
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(EVCNet, self).__init__()
        
        # Branch 1: EfficientNetV2S (Local feature extraction)
        effnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.efficientnet = nn.Sequential(
            effnet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # EfficientNetV2S output dimension is 1280
        
        # Branch 2: Vision Transformer (Global context extraction)
        # Using a tiny ViT variant with patch size 16 and embedding dimension 192 to match the paper
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        # ViT output dimension is 192
        
        # Feature Fusion Dimension = 1280 + 192 = 1472
        self.fusion_dim = 1472
        
        # Capsule Network Classifier
        self.primary_capsules = PrimaryCapsule(in_features=self.fusion_dim, num_capsules=32, dim_capsule=8)
        self.class_capsules = ClassCapsule(num_routes=32, in_dim=8, num_classes=num_classes, out_dim=16, routing_iters=3)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Extract features
        f_E = self.efficientnet(x) # (batch_size, 1280)
        f_V = self.vit(x)          # (batch_size, 192)
        
        # Feature Fusion
        f_fused = torch.cat((f_E, f_V), dim=1) # (batch_size, 1472)
        f_fused = self.dropout(f_fused)
        
        # Capsule Network processing
        u = self.primary_capsules(f_fused)
        v = self.class_capsules(u)
        
        # Calculate vector lengths as class probabilities
        classes = (v ** 2).sum(dim=-1) ** 0.5
        
        # Apply softmax to normalize probabilities to sum to 1
        probs = F.softmax(classes, dim=-1)
        
        return probs