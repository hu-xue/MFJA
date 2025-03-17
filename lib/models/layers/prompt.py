import torch
from torch import nn
import timm
import math
from functools import partial


'''
def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
'''


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class Bi_prompt(nn.Module):
    def __init__(self, inplanes, hide_channel, smooth=False):
        super().__init__()
        self.conv0_0 = nn.Conv2d(inplanes, hide_channel,1,1,0)
        self.conv0_1 = nn.Conv2d(inplanes, hide_channel,1,1,0)
        self.conv1_1 = nn.Conv2d(hide_channel, inplanes,1,1,0)
        self.fovea = Fovea(smooth)
        
        self.norm  = nn.LayerNorm(768, eps=1e-6)

    def forward(self, modal1, modal2):  # modal1 is H, modal2 is P, result is P
        z = modal1[:,:64,:]
        x = modal1[:,64:,:]
        
        zi = modal2[:,:64,:]
        xi = modal2[:,64:,:]
        
        z_feat = token2feature(self.norm(z))
        x_feat = token2feature(self.norm(x))
        zi_feat = token2feature(self.norm(zi))
        xi_feat = token2feature(self.norm(xi))
        
        z_feat = torch.cat([z_feat, zi_feat], dim=1)
        x_feat = torch.cat([x_feat, xi_feat], dim=1)
        
        B, C, W, H = z_feat.shape
        z0 = z_feat[:,0:int(C/2),:,:].contiguous()
        z0 = self.conv0_0(z0)
        z1 = z_feat[:,int(C/2):,:,:].contiguous()
        z1 = self.conv0_1(z1)
        z0 = self.fovea(z0) + z1
        z0 = self.conv1_1(z0)
        
        B, C, W, H = x_feat.shape
        x0 = x_feat[:,0:int(C/2),:,:].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x_feat[:,int(C/2):,:,:].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1
        x0 = self.conv1_1(x0)
        
        z0 = feature2token(z0)
        x0 = feature2token(x0)
        
        output = torch.cat([z0, x0], dim=1)
        
        return output

def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x
def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens
class Fovea(nn.Module):
    def __init__(self, smooth=False):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


"""


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        #print(x.shape)
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        #print(x_down.shape)

        x_patch = x_down[:, 64:].reshape(B, 16, 16, self.dim).permute(0, 3, 1, 2)   ############
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 16 * 16, self.dim)


        #x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up
"""