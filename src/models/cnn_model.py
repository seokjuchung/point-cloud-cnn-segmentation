import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes=5, input_channels=4):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Encoder (Downsampling path)
        self.enc1 = self._make_encoder_block(input_channels, 32)
        self.enc2 = self._make_encoder_block(32, 64)
        self.enc3 = self._make_encoder_block(64, 128)
        self.enc4 = self._make_encoder_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(256, 512)
        
        # Decoder (Upsampling path)
        self.dec4 = self._make_decoder_block(512, 256)
        self.dec3 = self._make_decoder_block(512, 128)  # 512 due to skip connection
        self.dec2 = self._make_decoder_block(256, 64)   # 256 due to skip connection
        self.dec1 = self._make_decoder_block(128, 32)   # 128 due to skip connection
        
        # Final classification layer
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)  # 64 due to skip connection
        
        # Dropout for regularization
        self.dropout = nn.Dropout3d(p=0.3)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path with skip connections
        enc1_out = self.enc1(x)
        enc1_pool = F.max_pool3d(enc1_out, kernel_size=2)
        
        enc2_out = self.enc2(enc1_pool)
        enc2_pool = F.max_pool3d(enc2_out, kernel_size=2)
        
        enc3_out = self.enc3(enc2_pool)
        enc3_pool = F.max_pool3d(enc3_out, kernel_size=2)
        
        enc4_out = self.enc4(enc3_pool)
        enc4_pool = F.max_pool3d(enc4_out, kernel_size=2)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_pool)
        bottleneck_out = self.dropout(bottleneck_out)
        
        # Decoder path with skip connections
        dec4_up = F.interpolate(bottleneck_out, scale_factor=2, mode='trilinear', align_corners=False)
        dec4_concat = torch.cat([dec4_up, enc4_out], dim=1)
        dec4_out = self.dec4(dec4_concat)
        
        dec3_up = F.interpolate(dec4_out, scale_factor=2, mode='trilinear', align_corners=False)
        dec3_concat = torch.cat([dec3_up, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_concat)
        
        dec2_up = F.interpolate(dec3_out, scale_factor=2, mode='trilinear', align_corners=False)
        dec2_concat = torch.cat([dec2_up, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_concat)
        
        dec1_up = F.interpolate(dec2_out, scale_factor=2, mode='trilinear', align_corners=False)
        dec1_concat = torch.cat([dec1_up, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_concat)
        
        # Final classification
        output = self.final_conv(dec1_out)
        
        return output