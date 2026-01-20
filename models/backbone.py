# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from H-DETR for Swin-T (https://github.com/HDETR/H-Deformable-DETR)
"""
Backbone modules.
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import SwinTransformer
from .vit import SimpleFeaturePyramid, ViT, LastLevelMaxPool

from .dinov2.models.vision_transformer import vit_large
from .dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class TransformerBackbone(nn.Module):
    def __init__(
        self, backbone: str, train_backbone: bool, return_interm_layers: bool, args
    ):
        super().__init__()
        out_indices = (1, 2, 3)
        if backbone == "swin_tiny":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 96
            # backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_small":
            backbone = SwinTransformer(
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 96
            # backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_base":
            backbone = SwinTransformer(
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 128
            # backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_large":
            backbone = SwinTransformer(
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=7,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 192
            # backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_large_window12":
            backbone = SwinTransformer(
                pretrain_img_size=384,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                ape=False,
                drop_path_rate=args.drop_path_rate,
                patch_norm=True,
                use_checkpoint=True,
                out_indices=out_indices,
            )
            embed_dim = 192
            # backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "vit_base":
            backbone = SimpleFeaturePyramid(
                net=ViT(
                    img_size=1024,
                    patch_size=16,
                    embed_dim=768,
                    depth=12,
                    num_heads=12,
                    drop_path_rate=0.1,
                    window_size=14,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
                    residual_block_indexes=[],
                    use_checkpoint=True,
                    use_rel_pos=True,
                    out_feature="last_feat",
                ),
                in_feature="last_feat",
                out_channels=256,
                # scale_factors=(4.0, 2.0, 1.0, 0.5),
                scale_factors=(2.0, 1.0, 0.5),
                # top_block=LastLevelMaxPool(),
                norm="LN",
                square_pad=1024,
            )
            embed_dim = 256
        elif backbone == "vit_large":
            window_block_indexes = []
            for i in range(24):
                if i % 6 != 5:
                    window_block_indexes.append(i)
            backbone = SimpleFeaturePyramid(
                net=ViT(
                    img_size=1024,
                    patch_size=16,
                    embed_dim=1024,
                    depth=24,
                    num_heads=16,
                    drop_path_rate=0.4,
                    window_size=14,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    window_block_indexes=window_block_indexes,
                    residual_block_indexes=[],
                    use_checkpoint=True,
                    use_rel_pos=True,
                    out_feature="last_feat",
                ),
                in_feature="last_feat",
                out_channels=256,
                # scale_factors=(4.0, 2.0, 1.0, 0.5),
                scale_factors=(2.0, 1.0, 0.5),
                # top_block=LastLevelMaxPool(),
                norm="LN",
                square_pad=1024,
            )
            embed_dim = 256
        else:
            raise NotImplementedError
        # backbone.init_weights(args.pretrained_backbone_path)

        for name, parameter in backbone.named_parameters():
            # TODO: freeze some layers?
            if not train_backbone:
                parameter.requires_grad_(False)

        if return_interm_layers:
            if 'vit' in args.backbone:
                self.strides = [16] * 3
                self.num_channels = [embed_dim] * 3
            else:
                self.strides = [8, 16, 32]
                self.num_channels = [
                    embed_dim * 2,
                    embed_dim * 4,
                    embed_dim * 8,
                ]
        else:
            if 'vit' in args.backbone:
                self.strides = [16]
                self.num_channels = [embed_dim]
            else:
                self.strides = [32]
                self.num_channels = [embed_dim * 8]

        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class DinoV2Backbone(nn.Module):
    def __init__(self, model_name="vit_large", pretrained=True, return_interm_layers=False):
        super().__init__()
        # 加载 dinov2 backbone
        self.backbone = dinov2_vitl14_reg(weights={'LVD142M': './pretrained/dinov2_vitl4_reg4_pretrain.pth'})

        self.embed_dim = self.backbone.embed_dim  # e.g., 1024 or 1152
        self.patch_size = self.backbone.patch_size  # usually 14
        self.return_interm_layers = return_interm_layers

        # decide how many scales to return
        if return_interm_layers:
            self.num_feature_levels = 3
            self.return_layers = [12, 18, 24]  # pick layers at different depths
            self.strides = [self.patch_size, self.patch_size * 2, self.patch_size * 4]  # 自定义
            self.num_channels = [self.embed_dim] * self.num_feature_levels
        else:
            self.num_feature_levels = 1
            self.return_layers = [24]  # only last layer
            self.strides = [self.patch_size]
            self.num_channels = [self.embed_dim]

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)  # ImageNet mean
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)    # ImageNet std

    def forward(self, tensor_list: NestedTensor):
        # x = tensor_list.tensors  # B, 3, H, W
        # mask = tensor_list.mask  # B, H, W
        x = F.interpolate(tensor_list.tensors, size=(224, 224), mode="bilinear", align_corners=False)  # B, 3, 224, 224
        mask = F.interpolate(tensor_list.mask[None].float(), size=(224, 224)).to(torch.bool)[0]  # B, 224, 224

        B, _, H, W = x.shape

        # 2. Normalize
        pixel_mean = self.pixel_mean.to(x.device)
        pixel_std = self.pixel_std.to(x.device)
        x = (x - pixel_mean) / pixel_std

        # Forward through DinoV2 backbone
        features_list = self.backbone.get_intermediate_layers(
            x, n=len(self.return_layers), return_class_token=False
        )  # list of (B, N, C)

        out = {}

        for idx, features in enumerate(features_list):
            # features: (B, N, C)
            patch_h = H // self.patch_size
            patch_w = W // self.patch_size

            # reshape back to feature map
            feat = features.permute(0, 2, 1).contiguous()  # (B, C, N)
            feat = feat.view(B, self.embed_dim, patch_h, patch_w)  # (B, C, H', W')

            # downsample if needed
            if idx == 1:
                feat = F.avg_pool2d(feat, kernel_size=2)  # Down x2
            elif idx == 2:
                feat = F.avg_pool2d(feat, kernel_size=4)  # Down x4

            m = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out[str(idx)] = NestedTensor(feat, m)

        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if "resnet" in args.backbone:
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation,
        )
    elif args.backbone.startswith("dinov2"):
        backbone = DinoV2Backbone(model_name=args.backbone, return_interm_layers=return_interm_layers)
    elif "swin" in args.backbone or "vit" in args.backbone:
        backbone = TransformerBackbone(
            args.backbone, train_backbone, return_interm_layers, args
        )
    model = Joiner(backbone, position_embedding)
    return model
