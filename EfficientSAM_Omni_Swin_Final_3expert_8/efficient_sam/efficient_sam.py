# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from .efficient_sam_decoder import MaskDecoder, PromptEncoder
from .efficient_sam_encoder import ImageEncoderViT
from .two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer
import monai

in_place = True

class SelfAttentionModule(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super(SelfAttentionModule, self).__init__()

        # Linear layers for query, key, and value projections
        self.norm1 = nn.BatchNorm1d(feature_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)

        self.norm2 = nn.BatchNorm1d(feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def self_attention(self, x):
        # Step 1: Multi-Head Self-Attention with Add & Norm (BatchNorm)
        # x shape: [batch_size, sequence_length, embed_dim]

        # Prepare the input for nn.MultiheadAttention (requires [sequence_length, batch_size, embed_dim])
        x_transposed = x.permute(1, 0, 2)  # [sequence_length, batch_size, embed_dim]

        # Multi-Head Self-Attention
        attn_output, _ = self.multihead_attn(x_transposed, x_transposed, x_transposed)

        # Transpose back to original shape
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, sequence_length, embed_dim]

        # Add & Norm (BatchNorm)
        x = x + self.dropout(attn_output)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)  # BatchNorm1d requires [batch_size, embed_dim, sequence_length]

        # Step 2: Feed-Forward Network with Add & Norm (BatchNorm)
        # Feed-Forward Network
        ffn_output = self.fc(x)

        # Add & Norm (BatchNorm)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)  # BatchNorm1d requires [batch_size, embed_dim, sequence_length]
        return x

    def forward(self, x):
        return self.self_attention(x)



class EfficientSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

        self.cls_token_decoder_MLP = nn.Conv2d(384,256,kernel_size=1, stride=1, padding=0)
        self.sls_token_decoder_MLP = nn.Conv2d(384,256,kernel_size=1, stride=1, padding=0)

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            # conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, bias=False)
        )

        self.fusionConv_swin = nn.Sequential(
            nn.GroupNorm(16, 384),
            nn.ReLU(inplace=in_place),
            # conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
            nn.Conv2d(384, 256, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, bias=False)
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        self.controller = nn.Conv2d(256 + 384 + 384, 162, kernel_size=1, stride=1, padding=0)
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            # nn.Conv3d(32, 8, kernel_size=1)
            nn.Conv2d(32, 8, kernel_size=(1, 1))
        )

        self.upsamplex2 = nn.Upsample(scale_factor=(2, 2))
        self.x1_resb_efficientsam = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))
        self.x2_resb_efficientsam = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))


        self.cls_token_encoder_MLP = nn.Conv2d(384,32 + 32 + 64 + 128 + 256, kernel_size=1, stride=1, padding=0)
        self.sls_token_encoder_MLP = nn.Conv2d(384,32 + 32 + 64 + 128 + 256, kernel_size=1, stride=1, padding=0)

        layers = [1, 2, 2, 2, 2]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=(1, 1), dilation=1, bias=False)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2))

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1))

        self.swin_backbone = monai.networks.nets.swin_unetr.SwinUNETR(img_size=512, in_channels=3, out_channels=8, spatial_dims=2)
        self.swin_resb = nn.Sequential(
            nn.GroupNorm(12, 24),
            nn.ReLU(inplace=in_place),
            # conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
            nn.Conv2d(24, 32, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1, bias=False)
        )

        self.mha_expert = SelfAttentionModule(feature_dim=256)


    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                # conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                #           weight_std=self.weight_std),
                nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=stride, padding=(0, 0), dilation=1,
                          bias=False)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=False))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=False))

        return nn.Sequential(*layers)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


    @torch.jit.export
    def predict_masks(
        self,
        ori_img: torch.Tensor,
        image_embeddings: torch.Tensor,
        # batched_points: torch.Tensor,
        # batched_point_labels: torch.Tensor,
        cls_token: torch.Tensor,
        sls_token: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        'PrPSeg_part'
        now_cls_token = cls_token.squeeze(0).repeat(ori_img.shape[0], 1).unsqueeze(-1).unsqueeze(-1)
        now_sls_token = sls_token.squeeze(0).repeat(ori_img.shape[0], 1).unsqueeze(-1).unsqueeze(-1)

        now_cls_token_PrPSeg = self.cls_token_encoder_MLP(now_cls_token)
        now_sls_token_PrPSeg = self.sls_token_encoder_MLP(now_sls_token)

        split_sizes = [32, 32, 64, 128, 256]
        cls_token_split = torch.split(now_cls_token_PrPSeg, split_sizes, dim=-3)
        sls_token_split = torch.split(now_sls_token_PrPSeg, split_sizes, dim=-3)

        x_omni = self.conv1(ori_img)

        x_omni = self.layer0(x_omni + cls_token_split[0] + sls_token_split[0])
        skip0 = x_omni

        x_omni = self.layer1(x_omni + cls_token_split[1] + sls_token_split[1])
        skip1 = x_omni

        x_omni = self.layer2(x_omni + cls_token_split[2] + sls_token_split[2])
        skip2 = x_omni

        x_omni = self.layer3(x_omni + cls_token_split[3] + sls_token_split[3])
        skip3 = x_omni

        x_omni = self.layer4(x_omni + cls_token_split[4] + sls_token_split[4])

        x_omni = self.fusionConv(x_omni)

        x_omni_feat = self.GAP(x_omni)

        'swin_backbone'
        ori_img2 = ori_img.detach().clone()
        # if not torch.jit.is_scripting():
        #     self.swin_backbone._check_input_size(ori_img2.shape[2:])

        hidden_states_out = self.swin_backbone.swinViT(ori_img2, self.swin_backbone.normalize)
        enc0 = self.swin_backbone.encoder1(ori_img2)
        enc1 = self.swin_backbone.encoder2(hidden_states_out[0])
        enc2 = self.swin_backbone.encoder3(hidden_states_out[1])
        enc3 = self.swin_backbone.encoder4(hidden_states_out[2])
        dec4 = self.swin_backbone.encoder10(hidden_states_out[4])

        x_swin = self.fusionConv_swin(dec4)
        x_swin_feat = self.GAP(x_swin)

        'EfficienSAM'
        x_efficientsam = self.fusionConv(image_embeddings)
        x_efficientsam_feat = self.GAP(x_efficientsam)


        'MoE attention'
        output = self.mha_expert(torch.cat([x_omni_feat, x_efficientsam_feat, x_swin_feat],2).squeeze(-1).permute([0,2,1]))
        moe_feat = (output[:,0] + output[:,1] + output[:,2]).unsqueeze(-1).unsqueeze(-1)

        x_cond = torch.cat([moe_feat, cls_token.unsqueeze(0).permute([0, 3, 1, 2]).repeat(ori_img.shape[0], 1, 1, 1), sls_token.unsqueeze(0).permute([0, 3, 1, 2]).repeat(ori_img.shape[0], 1, 1, 1)], 1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1)

        'EfficientSAM decoder'
        cls_emb = self.cls_token_decoder_MLP(cls_token.unsqueeze(0).permute([0,3,1,2])).permute([0,2,3,1]).squeeze(1)
        sls_emb = self.sls_token_decoder_MLP(sls_token.unsqueeze(0).permute([0,3,1,2])).permute([0,2,3,1]).squeeze(1)

        toekn_embeddings = torch.cat([cls_emb, sls_emb], 1)

        _, upscaled_embedding, iou_predictions = self.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=toekn_embeddings,
            multimask_output=False,
        )

        x_efficientsam_de = self.upsamplex2(upscaled_embedding)
        x_efficientsam_de = self.x2_resb_efficientsam(x_efficientsam_de)
        x_efficientsam_de = self.upsamplex2(x_efficientsam_de)
        x_efficientsam_de = self.x1_resb_efficientsam(x_efficientsam_de)

        'swin_decoder'
        dec3 = self.swin_backbone.decoder5(dec4, hidden_states_out[3])
        dec2 = self.swin_backbone.decoder4(dec3, enc3)
        dec1 = self.swin_backbone.decoder3(dec2, enc2)
        dec0 = self.swin_backbone.decoder2(dec1, enc1)
        out = self.swin_backbone.decoder1(dec0, enc0)
        swin_out = self.swin_resb(out)

        'PrPSeg decoder'
        x_omni = self.upsamplex2(x_omni)
        x_omni = x_omni + skip3
        x_omni = self.x8_resb(x_omni)

        # x4
        x_omni = self.upsamplex2(x_omni)
        x_omni = x_omni + skip2
        x_omni = self.x4_resb(x_omni)

        # x2
        x_omni = self.upsamplex2(x_omni)
        x_omni = x_omni + skip1
        x_omni = self.x2_resb(x_omni)

        # x1
        x_omni = self.upsamplex2(x_omni)
        x_omni = x_omni + skip0
        x_omni = self.x1_resb(x_omni)  # (32, 128, 256)

        head_inputs = self.precls_conv(x_omni+x_efficientsam_de+swin_out)
        N, _, H, W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)
        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)
        logits = logits.reshape(-1, 2, H, W)

        return logits

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    @torch.jit.export
    def get_image_embeddings(self, batched_images, task_id, scale_id) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images, task_id, scale_id)

    def forward(
        self,
        batched_images: torch.Tensor,
        task_id: torch.Tensor,
        scale_id: torch.Tensor,
        # batched_points: torch.Tensor,
        # batched_point_labels: torch.Tensor,
        scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        batch_size, _, input_h, input_w = batched_images.shape
        ori_img, image_embeddings, cls_token, sls_token = self.get_image_embeddings(batched_images, task_id, scale_id)
        return self.predict_masks(
            ori_img,
            image_embeddings,
            # batched_points,
            # batched_point_labels,
            cls_token = cls_token,
            sls_token = sls_token,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_efficient_sam(encoder_patch_embed_dim, encoder_num_heads, task_num, scale_num, checkpoint=None):
    img_size = 512   #ori 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
        task_num=task_num,
        scale_num=scale_num,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = EfficientSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict["model"], strict = False)
        print('SAM load successfully')
    return sam

class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes)
        # self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), stride=stride, padding=(1,1),dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(16, planes)
        # self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
        #                         dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=(1,1),dilation=1, bias=False)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out
