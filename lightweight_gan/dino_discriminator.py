import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math

class DINOv2Discriminator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        dino_model_name = 'dinov2_vitb14',  # Can be 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
        freeze_dino = True,
        use_cls_token = True,
        use_patch_tokens = True,
        pool_patch_tokens = 'mean',  # 'mean', 'max', or 'none'
    ):
        super().__init__()
        
        # Import DINOv2
        try:
            import torch.hub as hub
            # Load pretrained DINOv2 model
            self.dino = hub.load('facebookresearch/dinov2', dino_model_name)
            
            # Get feature dimension based on model variant
            model_dims = {
                'dinov2_vits14': 384,
                'dinov2_vitb14': 768,
                'dinov2_vitl14': 1024,
                'dinov2_vitg14': 1536,
            }
            self.dino_feature_dim = model_dims.get(dino_model_name, 768)
            
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            print("Falling back to manual initialization. Make sure to have the DINOv2 weights.")
            raise e
        
        # Freeze DINOv2 if specified
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()  # Set to eval mode
        
        self.freeze_dino = freeze_dino
        self.use_cls_token = use_cls_token
        self.use_patch_tokens = use_patch_tokens
        self.pool_patch_tokens = pool_patch_tokens
        self.image_size = image_size
        
        # Calculate total feature dimension
        total_feature_dim = 0
        if use_cls_token:
            total_feature_dim += self.dino_feature_dim
        if use_patch_tokens:
            if pool_patch_tokens != 'none':
                total_feature_dim += self.dino_feature_dim
            else:
                # For DINOv2 with 14x14 patches on 224x224 images
                # Adjust based on your image size
                num_patches = (image_size // 14) ** 2
                total_feature_dim += self.dino_feature_dim * num_patches
        
        # Input preprocessing for DINOv2 (expects 224x224 images)
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # Channel adaptation if needed
        if transparent:
            self.input_channels = 4
        elif greyscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
            
        # Convert to 3 channels if needed (DINOv2 expects RGB)
        if self.input_channels != 3:
            self.channel_adapter = nn.Conv2d(self.input_channels, 3, 1)
        else:
            self.channel_adapter = nn.Identity()
        
        # Multi-scale feature extraction heads
        # We'll use different projections for different purposes
        
        # Global discrimination head (real vs fake)
        self.global_head = nn.Sequential(
            nn.Linear(total_feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        # Patch-level discrimination head (for more detailed discrimination)
        if use_patch_tokens and pool_patch_tokens == 'none':
            self.patch_head = nn.Sequential(
                nn.Conv1d(self.dino_feature_dim, 256, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(256, 128, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(128, 1, 1)
            )
        else:
            self.patch_head = None
        
        # Auxiliary 32x32 discrimination head (maintaining compatibility)
        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(512, 1, 4)
        )
        
        # Simple decoder for auxiliary reconstruction loss
        self.decoder = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size * image_size * self.input_channels),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.global_head, self.patch_head, self.to_shape_disc_out, self.decoder]:
            if m is not None:
                for layer in m.modules():
                    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                        nn.init.kaiming_normal_(layer.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.constant_(layer.weight, 1)
                        nn.init.constant_(layer.bias, 0)
    
    def extract_dino_features(self, x):
        """Extract features from DINOv2 model"""
        # Adapt channels if necessary
        x = self.channel_adapter(x)
        
        # Resize to 224x224 for DINOv2
        x_resized = self.resize(x)
        
        # Normalize for DINOv2 (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_normalized = (x_resized - mean) / std
        
        # Extract features
        if self.freeze_dino:
            with torch.no_grad():
                # Get intermediate features
                features_dict = self.dino.forward_features(x_normalized)
        else:
            features_dict = self.dino.forward_features(x_normalized)
        
        # Extract different types of features
        features = []
        
        # Handle different DINOv2 output formats
        if isinstance(features_dict, dict):
            # Newer versions return a dictionary
            if self.use_cls_token and 'x_norm_clstoken' in features_dict:
                cls_token = features_dict['x_norm_clstoken']
                features.append(cls_token)
            elif self.use_cls_token:
                # Fallback: extract CLS token from full sequence
                full_features = features_dict.get('x_norm_patchtokens', features_dict.get('x', None))
                if full_features is not None:
                    cls_token = full_features[:, 0]
                    features.append(cls_token)
            
            if self.use_patch_tokens and 'x_norm_patchtokens' in features_dict:
                patch_tokens = features_dict['x_norm_patchtokens']
                if self.pool_patch_tokens == 'mean':
                    patch_features = patch_tokens.mean(dim=1)
                elif self.pool_patch_tokens == 'max':
                    patch_features = patch_tokens.max(dim=1)[0]
                else:  # 'none'
                    patch_features = patch_tokens.flatten(1)
                features.append(patch_features)
        else:
            # Older versions or different format
            # Assuming features_dict is the full feature tensor [B, N, D]
            if self.use_cls_token:
                cls_token = features_dict[:, 0]
                features.append(cls_token)
            
            if self.use_patch_tokens:
                patch_tokens = features_dict[:, 1:]  # Skip CLS token
                if self.pool_patch_tokens == 'mean':
                    patch_features = patch_tokens.mean(dim=1)
                elif self.pool_patch_tokens == 'max':
                    patch_features = patch_tokens.max(dim=1)[0]
                else:  # 'none'
                    patch_features = patch_tokens.flatten(1)
                features.append(patch_features)
        
        # Concatenate all features
        if len(features) > 0:
            combined_features = torch.cat(features, dim=-1)
        else:
            raise ValueError("No features extracted from DINOv2")
        
        return combined_features, patch_tokens if (self.use_patch_tokens and self.pool_patch_tokens == 'none') else None
    
    def forward(self, x, calc_aux_loss=False):
        orig_img = x
        batch_size = x.shape[0]
        
        # Extract DINOv2 features
        dino_features, patch_tokens = self.extract_dino_features(x)
        
        # Global discrimination
        out = self.global_head(dino_features).flatten(1)
        
        # Patch-level discrimination (if applicable)
        if self.patch_head is not None and patch_tokens is not None:
            # Reshape patch tokens for conv1d: [B, N, D] -> [B, D, N]
            patch_tokens = patch_tokens.transpose(1, 2)
            patch_out = self.patch_head(patch_tokens)  # [B, 1, N]
            # Average over patches
            patch_score = patch_out.mean(dim=-1).flatten(1)
            # Combine with global score
            out = out + 0.5 * patch_score
        
        # 32x32 auxiliary discrimination (for training stability)
        img_32x32 = F.interpolate(orig_img, size=(32, 32), mode='bilinear', align_corners=False)
        # Ensure 3 channels for the auxiliary head
        if self.input_channels != 3:
            img_32x32 = self.channel_adapter(img_32x32)
        out_32x32 = self.to_shape_disc_out(img_32x32)
        
        # Calculate auxiliary loss if requested
        aux_loss = None
        if calc_aux_loss:
            # Reconstruction loss using DINOv2 features
            reconstructed = self.decoder(dino_features)
            reconstructed = reconstructed.view(batch_size, self.input_channels, self.image_size, self.image_size)
            
            # Downsample original image to match reconstruction
            target = F.interpolate(orig_img, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            aux_loss = F.mse_loss(reconstructed, target)
        
        return out, out_32x32, aux_loss


# Alternative: Hybrid Discriminator that combines CNN and DINOv2
class HybridDINOv2Discriminator(nn.Module):
    """
    A hybrid discriminator that combines traditional CNN features with DINOv2 features
    for more robust discrimination.
    """
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        dino_model_name = 'dinov2_vitb14',
        freeze_dino = True,
        cnn_weight = 0.5,  # Balance between CNN and DINOv2 features
    ):
        super().__init__()
        
        self.cnn_weight = cnn_weight
        self.dino_weight = 1.0 - cnn_weight
        
        # Import the original CNN-based layers from the file
        from lightweight_gan import SPConvDownsample, SumBranches, Blur, PreNorm, LinearAttention
        
        # Initialize DINOv2
        try:
            import torch.hub as hub
            self.dino = hub.load('facebookresearch/dinov2', dino_model_name)
            
            model_dims = {
                'dinov2_vits14': 384,
                'dinov2_vitb14': 768,
                'dinov2_vitl14': 1024,
                'dinov2_vitg14': 1536,
            }
            self.dino_feature_dim = model_dims.get(dino_model_name, 768)
            
        except Exception as e:
            print(f"Error loading DINOv2: {e}")
            raise e
        
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
        
        self.freeze_dino = freeze_dino
        self.image_size = image_size
        
        # Setup input channels
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3
        
        self.input_channels = init_channel
        
        # Channel adapter for DINOv2
        if self.input_channels != 3:
            self.channel_adapter = nn.Conv2d(self.input_channels, 3, 1)
        else:
            self.channel_adapter = nn.Identity()
        
        # Resize for DINOv2
        self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # CNN pathway (simplified version of original discriminator)
        resolution = int(math.log2(image_size))
        num_layers = min(resolution - 2, 6)  # Limit depth
        
        self.cnn_layers = nn.ModuleList([])
        
        channels = init_channel
        for i in range(num_layers):
            out_channels = min(channels * 2, fmap_max)
            
            layer = nn.Sequential(
                Blur() if i > 0 else nn.Identity(),
                nn.Conv2d(channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2)
            )
            
            self.cnn_layers.append(layer)
            channels = out_channels
        
        self.cnn_feature_dim = channels
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + self.dino_feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
        
        # Auxiliary 32x32 head
        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(512, 1, 4)
        )
        
        # Simple decoder for auxiliary loss
        self.decoder = nn.Sequential(
            nn.Linear(self.cnn_feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_size * image_size * init_channel // 16),
            nn.Tanh()
        )
    
    def extract_cnn_features(self, x):
        """Extract CNN features"""
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return x
    
    def extract_dino_features(self, x):
        """Extract DINOv2 features"""
        x = self.channel_adapter(x)
        x_resized = self.resize(x)
        
        # Normalize for DINOv2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_normalized = (x_resized - mean) / std
        
        if self.freeze_dino:
            with torch.no_grad():
                features = self.dino(x_normalized)
        else:
            features = self.dino(x_normalized)
        
        return features
    
    def forward(self, x, calc_aux_loss=False):
        orig_img = x
        
        # Extract both CNN and DINOv2 features
        cnn_features = self.extract_cnn_features(x)
        dino_features = self.extract_dino_features(x)
        
        # Fuse features
        combined_features = torch.cat([
            cnn_features * self.cnn_weight,
            dino_features * self.dino_weight
        ], dim=1)
        
        # Main discrimination output
        out = self.fusion(combined_features).flatten(1)
        
        # 32x32 auxiliary output
        img_32x32 = F.interpolate(orig_img, size=(32, 32), mode='bilinear', align_corners=False)
        if self.input_channels != 3:
            img_32x32 = self.channel_adapter(img_32x32)
        out_32x32 = self.to_shape_disc_out(img_32x32)
        
        # Auxiliary loss
        aux_loss = None
        if calc_aux_loss:
            reconstructed = self.decoder(cnn_features)
            reconstructed = reconstructed.view(x.shape[0], self.input_channels, 
                                             self.image_size // 4, self.image_size // 4)
            target = F.interpolate(orig_img, size=reconstructed.shape[2:], mode='bilinear', align_corners=False)
            aux_loss = F.mse_loss(reconstructed, target)
        
        return out, out_32x32, aux_loss