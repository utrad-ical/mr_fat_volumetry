"""
3D FC-ResNet model by Myronenko A. (https://arxiv.org/pdf/1810.11654.pdf)
Created on Thu Aug 15 11:30:46 2019
@author: ynomura
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Registration3d(nn.Module):
    def __init__(self, inc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
            inc: input channel
            target: -1
        """
        super(Registration3d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.p_conv = nn.Conv3d(inc, 3*(inc-1), kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv3d(inc, inc-1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        #(b, 3*(inc-1), NSlice, height, width)
        x0 = x[:,-1:].detach()
        
        offset = self.p_conv(x)
        #(b, (inc-1), NSlice, height, width)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        
        dtype = offset.data.type()
        b, N, n, h, w = offset.size(0), offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)
        
        if self.padding:
            x = nn.functional.pad(x[:,:-1], (self.padding,self.padding,self.padding,self.padding,self.padding,self.padding))
            
        # (b, 3N, n, h, w)
        p = self._get_p(offset, dtype)
        
        # (b, N, n, h, w, 3)
        p = p.contiguous().view(b, 3, N, n, h, w).permute(0, 2, 3, 4, 5, 1)
        q = p.detach()
        
        q[..., 0] = torch.clamp(q[..., 0], 0, x.size(2)-2)
        q[..., 1] = torch.clamp(q[..., 1], 0, x.size(3)-2)
        q[..., 2] = torch.clamp(q[..., 2], 0, x.size(4)-2)
        p0 = q.floor()
        
        out = self.get_intensity(x, q[..., 0], q[..., 1], q[..., 2]) *\
              (1 + (p0[...,0] - q[...,0])) * (1 + (p0[...,1] - q[...,1])) * (1 + (p0[...,2] - q[...,2])) +\
              self.get_intensity(x, q[..., 0] + 1, q[..., 1], q[..., 2]) *\
              (  - (p0[...,0] - q[...,0])) * (1 + (p0[...,1] - q[...,1])) * (1 + (p0[...,2] - q[...,2])) +\
              self.get_intensity(x, q[..., 0], q[..., 1] + 1, q[..., 2]) *\
              (1 + (p0[...,0] - q[...,0])) * (  - (p0[...,1] - q[...,1])) * (1 + (p0[...,2] - q[...,2])) +\
              self.get_intensity(x, q[..., 0], q[..., 1], q[..., 2] + 1) *\
              (1 + (p0[...,0] - q[...,0])) * (1 + (p0[...,1] - q[...,1])) * (  - (p0[...,2] - q[...,2])) +\
              self.get_intensity(x, q[..., 0] + 1, q[..., 1] + 1, q[..., 2]) *\
              (  - (p0[...,0] - q[...,0])) * (  - (p0[...,1] - q[...,1])) * (1 + (p0[...,2] - q[...,2])) +\
              self.get_intensity(x, q[..., 0] + 1, q[..., 1], q[..., 2] + 1) *\
              (  - (p0[...,0] - q[...,0])) * (1 + (p0[...,1] - q[...,1])) * (  - (p0[...,2] - q[...,2])) +\
              self.get_intensity(x, q[..., 0], q[..., 1] + 1, q[..., 2] + 1) *\
              (1 + (p0[...,0] - q[...,0])) * (  - (p0[...,1] - q[...,1])) * (  - (p0[...,2] - q[...,2])) +\
              self.get_intensity(x, q[..., 0] + 1, q[..., 1] + 1, q[..., 2] + 1) *\
              (  - (p0[...,0] - q[...,0])) * (  - (p0[...,1] - q[...,1])) * (  - (p0[...,2] - q[...,2]))

        # modulation
        if self.modulation:
            out = out * m

        out = torch.cat([out, x0], dim=1)
        return out
    
    def _get_p_0(self, n, h, w, N, dtype):
        p_0_z, p_0_y, p_0_x = torch.meshgrid(
            torch.arange(1, n+1),
            torch.arange(1, h+1),
            torch.arange(1, w+1))
        p_0_x = torch.flatten(p_0_x).view(1, 1, n, h, w).repeat(1, N, 1, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, n, h, w).repeat(1, N, 1, 1, 1)
        p_0_z = torch.flatten(p_0_z).view(1, 1, n, h, w).repeat(1, N, 1, 1, 1)
        p_0 = torch.cat([p_0_z, p_0_y, p_0_x], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, n, h, w = offset.size(1)//3, offset.size(2), offset.size(3), offset.size(4)
        # (1, 3N, n, h, w)
        p_0 = self._get_p_0(n, h, w, N, dtype)
        p = p_0 + offset
        return p
    
    def get_intensity(self, x, p_z, p_y, p_x):
        
        b, N, n, h, w = p_z.size(0), p_z.size(1), p_z.size(2), p_z.size(3), p_z.size(4)
        padded_h, padded_w = x.size(3), x.size(4)
        x = x.contiguous().view(b, N, -1)
        
        index = p_z * padded_h * padded_w + p_y * padded_w + p_x
        index = index.contiguous().view(b, N, -1)
        return x.gather(dim=-1, index=index.detach().type(torch.long)).contiguous().view(b, N, n, h, w)


class _ReLUBnConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):

        super(_ReLUBnConv, self).__init__()
        padding_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size, stride,
                      padding=padding_size, bias=False))

    def forward(self, x):
        return self.conv(x)


class _ResBlockL(nn.Module):

    def __init__(self, in_channels, out_channels, parameter_n=2, down_sampling=False, up_sampling=False):

        super(_ResBlockL, self).__init__()
        stride1 = 2 if down_sampling else 1
        intermediate_channels = (out_channels // parameter_n) if parameter_n <= out_channels else 1

        self.conv1 = _ReLUBnConv(in_channels, intermediate_channels, 1, stride1)
        self.conv2 = _ReLUBnConv(intermediate_channels, intermediate_channels, 3)
        self.conv3 = _ReLUBnConv(intermediate_channels, out_channels, 1)

        self.shortcut = nn.Sequential(
                           nn.Conv3d(in_channels, intermediate_channels,
                                     1, stride1, 0, bias=False),
                           nn.Conv3d(intermediate_channels, out_channels,
                                     1, 1, 0, bias=False))

        self.up2_base = None
        self.up2_shortcut = None
        if up_sampling:
            self.up2_base = nn.Upsample(scale_factor=2, mode='trilinear')
            self.up2_shortcut = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.up2_base is not None:
            out = self.up2_base(out)
            
        shortcut_out = self.shortcut(x)
        if self.up2_shortcut is not None:
            shortcut_out = self.up2_shortcut(shortcut_out)

        out += shortcut_out
        return out

class _ResBlockS(nn.Module):

    def __init__(self, in_channels, out_channels, down_sampling=False, up_sampling=False):
 
        super(_ResBlockS, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channels)

        self.down2_base = None
        self.down2_shortcut = None
        if down_sampling:
            self.down2_base = nn.MaxPool3d(kernel_size=2, stride=2)
            self.down2_shortcut = nn.MaxPool3d(kernel_size=2, stride=2) 

        self.conv_base = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv_shortcut = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False)
        
        self.up2_base = None
        self.up2_shortcut = None
        if up_sampling:
            self.up2_base = nn.Upsample(scale_factor=2, mode='trilinear')
            self.up2_shortcut = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x):

        out = self.relu(x)
        out = self.bn(out)

        if self.down2_base is not None:
            out = self.down2_base(out)
        
        out = self.conv_base(out)
        
        if self.up2_base is not None:
            out = self.up2_base(out)
            
        shortcut_out = x

        if self.down2_shortcut is not None:
            shortcut_out = self.down2_shortcut(shortcut_out)
        
        shortcut_out = self.conv_shortcut(shortcut_out)
        
        if self.up2_shortcut is not None:
            shortcut_out = self.up2_shortcut(shortcut_out)

        out += shortcut_out
        return out


class FC_RESNET(nn.Module):

    def __init__(self, in_channels, output_channels, parameter_n, registration=0):

        super(FC_RESNET, self).__init__()
        
        # Encoder
        
        if registration>0:
            self.enc_initial_conv = nn.Sequential(
                                    Registration3d(in_channels),
                                    nn.Conv3d(in_channels, 32, 3, 1, padding=1, bias=False))
        else:
            self.enc_initial_conv = nn.Conv3d(in_channels, 32, 3, 1, padding=1, bias=False)
        
        self.enc_block_s_1 = _ResBlockS(32, 32, down_sampling=True)
        self.enc_block_l_1 = _ResBlockL(32, 64, parameter_n=parameter_n, down_sampling=True)
        self.enc_block_l_2 = _ResBlockL(64, 128, parameter_n=parameter_n)
        self.enc_block_l_3 = _ResBlockL(128, 256, parameter_n=parameter_n)
        self.enc_block_l_4 = _ResBlockL(256, 256, parameter_n=parameter_n, down_sampling=True)
        self.enc_block_l_5 = _ResBlockL(256, 256, parameter_n=parameter_n)
        self.enc_block_l_6 = _ResBlockL(256, 256, parameter_n=parameter_n)
        self.enc_block_l_7 = _ResBlockL(256, 256, parameter_n=parameter_n)
        self.enc_block_l_8 = _ResBlockL(256, 256, parameter_n=parameter_n, down_sampling=True)

        # Decoder
        self.dec_block_l_1 = _ResBlockL(256, 128, parameter_n=parameter_n, up_sampling=True)
        self.dec_block_l_2 = _ResBlockL(128+256, 256, parameter_n=parameter_n)
        self.dec_block_l_3 = _ResBlockL(256, 128, parameter_n=parameter_n)
        self.dec_block_l_4 = _ResBlockL(128, 64, parameter_n=parameter_n)
        self.dec_block_l_5 = _ResBlockL(64, 32, parameter_n=parameter_n, up_sampling=True)
        self.dec_block_l_6 = _ResBlockL(32+256, 32, parameter_n=parameter_n)
        self.dec_block_l_7 = _ResBlockL(32, 16, parameter_n=parameter_n)
        self.dec_block_l_8 = _ResBlockL(16, 8, parameter_n=parameter_n, up_sampling=True)
        self.dec_block_s_1 = _ResBlockS(8+32, 8, up_sampling=True)
        if output_channels == 1:
            self.dec_final_convs = nn.Sequential(
                                   nn.Conv3d(8 + 32, 32, 3, 1, padding=1, bias=False),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, output_channels, 1, 1, 0, bias=False),
                                   nn.Sigmoid())
        else:
            self.dec_final_convs = nn.Sequential(
                                   nn.Conv3d(8 + 32, 32, 3, 1, padding=1, bias=False),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, output_channels, 1, 1, 0, bias=False),
                                   nn.Sigmoid())
                                   #nn.Softmax(dim=1))

    def forward(self, x):

        # Encoder
        enc1 = self.enc_initial_conv(x)
        enc2 = self.enc_block_s_1(enc1)
        enc3 = self.enc_block_l_1(enc2)
        enc4 = self.enc_block_l_2(enc3)
        enc5 = self.enc_block_l_3(enc4)
        enc6 = self.enc_block_l_4(enc5)
        enc7 = self.enc_block_l_5(enc6)
        enc8 = self.enc_block_l_6(enc7)
        enc9 = self.enc_block_l_7(enc8)
        enc10 = self.enc_block_l_8(enc9)
        # Decoder        
        
        dec1 = self.dec_block_l_1(enc10)
        if (dec1.shape[-3] > enc9.shape[-3])|(dec1.shape[-2] > enc9.shape[-2])|(dec1.shape[-1] > enc9.shape[-1]):
            dec1 = dec1[:, :, :enc9.shape[-3], :enc9.shape[-2], :enc9.shape[-1]]
        if (dec1.shape[-3] < enc9.shape[-3])|(dec1.shape[-2] < enc9.shape[-2])|(dec1.shape[-1] < enc9.shape[-1]):
            dec1 = F.pad(dec1, [0, enc9.shape[-1] - dec1.shape[-1],\
                                0, enc9.shape[-2] - dec1.shape[-2],\
                                0, enc9.shape[-3] - dec1.shape[-3]])
        dec1 = torch.cat((dec1, enc9), 1)
        dec2 = self.dec_block_l_2(dec1)
        dec3 = self.dec_block_l_3(dec2)
        dec4 = self.dec_block_l_4(dec3)
        dec5 = self.dec_block_l_5(dec4)
        if (dec5.shape[-3] > enc5.shape[-3])|(dec5.shape[-2] > enc5.shape[-2])|(dec5.shape[-1] > enc5.shape[-1]):
            dec5 = dec5[:, :, :enc5.shape[-3], :enc5.shape[-2], :enc5.shape[-1]]
        if (dec5.shape[-3] < enc5.shape[-3])|(dec5.shape[-2] < enc5.shape[-2])|(dec5.shape[-1] < enc5.shape[-1]):
            dec5 = F.pad(dec1, [0, enc5.shape[-1] - dec5.shape[-1],\
                                0, enc5.shape[-2] - dec5.shape[-2],\
                                0, enc5.shape[-3] - dec5.shape[-3]])
        dec5 = torch.cat((dec5, enc5), 1)    
        dec6 = self.dec_block_l_6(dec5)
        dec7 = self.dec_block_l_7(dec6)
        dec8 = self.dec_block_l_8(dec7)
        if (dec8.shape[-3] > enc2.shape[-3])|(dec8.shape[-2] > enc2.shape[-2])|(dec8.shape[-1] > enc2.shape[-1]):
            dec8 = dec8[:, :, :enc2.shape[-3], :enc2.shape[-2], :enc2.shape[-1]]
        if (dec8.shape[-3] < enc2.shape[-3])|(dec8.shape[-2] < enc2.shape[-2])|(dec8.shape[-1] < enc2.shape[-1]):
            dec8 = F.pad(dec8, [0, enc2.shape[-1] - dec8.shape[-1],\
                                0, enc2.shape[-2] - dec8.shape[-2],\
                                0, enc2.shape[-3] - dec8.shape[-3]])
        dec8 = torch.cat((dec8, enc2), 1)
        dec9 = self.dec_block_s_1(dec8)
        if (dec9.shape[-3] > enc1.shape[-3])|(dec9.shape[-2] > enc1.shape[-2])|(dec9.shape[-1] > enc1.shape[-1]):
            dec9 = dec9[:, :, :enc1.shape[-3], :enc1.shape[-2], :enc1.shape[-1]]
        if (dec9.shape[-3] < enc1.shape[-3])|(dec9.shape[-2] < enc1.shape[-2])|(dec9.shape[-1] < enc1.shape[-1]):
            dec9 = F.pad(dec9, [0, enc1.shape[-1] - dec9.shape[-1],\
                                0, enc1.shape[-2] - dec9.shape[-2],\
                                0, enc1.shape[-3] - dec9.shape[-3]])
        dec9 = torch.cat((dec9, enc1), 1)
        out = self.dec_final_convs(dec9)

        return out
