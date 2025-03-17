import torch
from torch import nn
import timm
import math


from lib.models.mfja.sfm import SFM
from lib.models.mfja.cfm import CFM_woGIM, GIM
from lib.utils.token_utils import token2patch, patch2token


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



class Iner_adapter(nn.Module):
    def __init__(self, dim=768, ratio=0.05, skip=False):
        super().__init__()
        self.skip = skip
        self.dim_mid = 8#int(dim * ratio)
        self.down = nn.Linear(dim, self.dim_mid)
        self.mid = nn.GELU()
        self.up = nn.Linear(self.dim_mid, dim)
        
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x):
        x_down = self.down(x)
        x_down = self.mid(x_down)
        x_up = self.up(x_down)
        if self.skip is False:
            x_up = x_up
        else:
            x_up = x + x_up
        return x_up


class Fusion_adapter(nn.Module):
    def __init__(self, dim=768, ratio=0.25):
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.mid_dim = 8  #int(dim * ratio)
        self.adapter_down = nn.Linear(dim, self.mid_dim)
        
        self.evi_layer = nn.Linear(self.mid_dim, self.mid_dim)
        
        self.adapter_mid = nn.GELU()
        self.adapter_up = nn.Linear(8, dim)
        
        self.dropout = nn.Dropout(0.1)
        
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        

    def forward(self, x):
        B, N, C = x.shape
        
        x_down = self.adapter_down(x)
        evi_x_down = self.evi_layer(x_down)
        evi_x_down = torch.nn.functional.softmax(evi_x_down, dim=-1)
        x_down = self.adapter_mid(x_down * evi_x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        
        return x_up