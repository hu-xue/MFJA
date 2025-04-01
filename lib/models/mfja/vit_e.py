import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_adapter

from ..layers.attn_adapt_blocks import CEABlock  
from ..layers.dualstream_attn_blocks import DSBlock ## Dual Stream without adapter



from lib.models.layers.adapter import Fusion_adapter

_logger = logging.getLogger(__name__)


# todo 
class VisionTransformerMFJA(VisionTransformer):
    """ Vision Transformer with MFJA module, created by xuehu. """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, adapter_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size 
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size  # patch size
        self.in_chans = in_chans  

        self.num_classes = num_classes 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """     #
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1
            if i < 20:
                blocks.append(
                CEABlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                        attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                        keep_ratio_search=ce_keep_ratio_i, layer_num=i)
                )

        self.blocks = nn.Sequential(*blocks)
        self.init_weights(weight_init)

       

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]  # batch, channel, height, width
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb  #! modal1
        xi, zi = x_dte, z_dte  #! modal2


        # print("input x",x.size())
        #! modal1
        x = self.patch_embed(x)
        z = self.patch_embed(z)

        # print("after patch_embed x",x.size())

        #! modal2
        xi = self.patch_embed(xi)
        zi = self.patch_embed(zi)
        
###################################################################===========
        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed
        # add position embedding, learned parameter
        z += self.pos_embed_z
        x += self.pos_embed_x

        zi += self.pos_embed_z
        xi += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
            xi += self.search_segment_pos_embed
            zi += self.template_segment_pos_embed
        #print(x.shape) #[Batch size, 256, 768] #! finished patch embedding
        
        x = combine_tokens(z, x, mode=self.cat_mode)  ## [Batch size, 64, 768] + [Batch size, 256, 768] = [Batch size, 320, 768]  #! modal1
        #print("after cat",x.shape)

        xi = combine_tokens(zi, xi, mode=self.cat_mode)  #! modal2
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            xi = torch.cat([cls_tokens, xi], dim=1)

        x = self.pos_drop(x)#! modal1
        xi = self.pos_drop(xi)#! modal2

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        global_index_ti = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_ti = global_index_ti.repeat(B, 1)

        global_index_si = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_si = global_index_si.repeat(B, 1)

        removed_indexes_s = []
        removed_indexes_si = []

        removed_flag = False
        for i, blk in enumerate(self.blocks):
                
            x, global_index_t, global_index_s, removed_index_s, attn, \
            xi, global_index_ti, global_index_si, removed_index_si, attn_i = \
                blk(x, xi, global_index_t, global_index_ti, global_index_s, global_index_si, \
                    mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)
                removed_indexes_si.append(removed_index_si)

              
        x = self.norm(x)  #! normalize the output, shape: [Batch size, 320, 768]
        xi = self.norm(xi)

        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]
        lens_xi_new = global_index_si.shape[1]
        lens_zi_new = global_index_ti.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]
        zi = xi[:, :lens_zi_new]
        xi = xi[:, lens_zi_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        
        if removed_indexes_si and removed_indexes_si[0] is not None:
            removed_indexes_cat_i = torch.cat(removed_indexes_si, dim=1)

            pruned_lens_xi = lens_x - lens_xi_new                                ########################
            pad_xi = torch.zeros([B, pruned_lens_xi, xi.shape[2]], device=xi.device)
            xi = torch.cat([xi, pad_xi], dim=1)
            index_all = torch.cat([global_index_si, removed_indexes_cat_i], dim=1)
            # recover original token order
            C = xi.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            xi = torch.zeros_like(xi).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=xi)
        
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)  # recover the original token order
        xi = recover_tokens(xi, lens_zi_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)  # shape: [Batch size, 320, 768]
        xi = torch.cat([zi, xi], dim=1)  # shape: [Batch size, 320, 768]
        #x = torch.cat([x, xi], dim=0)
        #print("===========final out: ",x.size())
        
        x = x + xi

        #x = torch.cat([x, xi], dim=2)
        #x = self.adap_headcat(x)


        #print("-------",x.shape)
        #print("attn",attn.size())
        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerMFJA(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_e(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929)."""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


'''
class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(CMA_Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channel, hidden_channel, kernel_size=1, stride=1, padding=0
        )

        self.scale = hidden_channel ** -0.5

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                hidden_channel, out_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, rgb, freq):  # b,c,h,w  b,c,h,w
        _, _, h, w = rgb.size()

        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)

        q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3)).transpose(
            -2, -1
        )
        k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))

        attn = torch.matmul(q, k) * self.scale
        m = attn.softmax(dim=-1)

        v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3)).transpose(
            -2, -1
        )
        z = torch.matmul(m, v)
        z = z.view(z.size(0), h, w, -1)
        z = z.permute(0, 3, 1, 2).contiguous()

        output = rgb + self.conv4(z)

        return output
'''
class Prompt_block(nn.Module, ):
    '''Prompt block: inplanes is embed_dim, hide_channel is 8, smooth is true.
    '''
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1_1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea1 = Fovea(smooth=smooth)
        self.fovea2 = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. 0-dim/2 is the rgb modal, dim/2-1 is the thermal modal."""
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous() # [B, C/2, W, H]  rgb template
        x0 = self.conv0_0(x0)  # [B, hide_channel, W, H]
        x1 = x[:, int(C/2):, :, :].contiguous()  # [B, C/2, W, H]  thermal template
        x1 = self.conv0_1(x1)
        
        x0_out = self.fovea1(x0) + x1  # x0_out = M_thermal * M_fovea + M_RGB
        x0_out = self.conv1x1_1(x0_out)
        
        x1_out = self.fovea2(x1) + x0  # x1_out = M_RGB * M_fovea + M_thermal
        x1_out = self.conv1x1_2(x1_out)
        
        out = torch.cat([x0_out, x1_out], dim=1)  # 
        return out
    
    
class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)  # lambda * M_RGB = M_fovea
        else:
            mask = self.softmax(x)
        output = mask * x  # M_RGB * M_fovea
        output = output.contiguous().view(b, c, h, w)

        return output
    
    
def candidate_elimination_prompt(tokens: torch.Tensor, lens_t: int, global_index: torch.Tensor):
    # tokens: actually prompt_parameters
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=global_index.unsqueeze(-1).expand(B, -1, C))
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new


def multimodal_prompt():
    pass

# def vit_large_patch16_224_ce_adapter(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
#     """
#     model_kwargs = dict(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
#     return model
