from types import SimpleNamespace

import torch
import torch.nn as nn

from lxmert.src.lxrt.entry import LXRTEncoder

use_cuda = torch.cuda.is_available()

class LXMERTOracleInputTarget(nn.Module):
    """docstring for Oracle"""
    def __init__(self, no_categories, no_category_feat, no_hidden_encoder,
                 mlp_layer_sizes, no_visual_feat, no_crop_feat, inputs_config,
                 lxmert_encoder_args, scale_visual_to=None):
        super(LXMERTOracleInputTarget, self).__init__()

        self.no_hidden_encoder      = no_hidden_encoder
        self.mlp_layer_sizes        = mlp_layer_sizes
        self.no_categories          = no_categories
        self.no_category_feat       = no_category_feat
        self.n_spatial_feat         = 8
        self.no_visual_feat         = no_visual_feat
        self.no_crop_feat           = no_crop_feat
        self.inputs_config          = inputs_config
        self.scale_visual_to        = scale_visual_to

        if self.scale_visual_to != 0:
            if type(self.scale_visual_to) == int:
                self.scale_crop = nn.Linear(self.no_crop_feat, scale_visual_to)
            elif type(self.scale_visual_to) == list:
                self.scale_crop = nn.Linear(self.no_crop_feat, scale_visual_to[1])

        self.no_mlp_inputs = 768

        self.mlp_layer_sizes = [self.no_mlp_inputs] + self.mlp_layer_sizes
        self.mlp = nn.Sequential()
        idx = 0
        for i in range(len(self.mlp_layer_sizes)-1):
            self.mlp.add_module(str(idx), nn.Linear(self.mlp_layer_sizes[i], self.mlp_layer_sizes[i+1]))
            idx += 1
            if i < len(mlp_layer_sizes)-1:
                self.mlp.add_module(str(idx), nn.ReLU())
                idx += 1
            else:
                 self.mlp.add_module(str(idx), nn.LogSoftmax(dim=-1))

        lxrt_encoder_args = SimpleNamespace(**lxmert_encoder_args)
        self.lxrt_encoder = LXRTEncoder(
            lxrt_encoder_args,
            max_seq_length=200
        )

        if not lxrt_encoder_args.from_scratch:
            print("Loading LXMERT pretrained model...")
            self.lxrt_encoder.load(lxrt_encoder_args.model_path)
        else:
            print("Initializing LXMERT model from scratch...")

    def forward(self, fasterrcnn_features, fasterrcnn_boxes, target_bbox, 
                input_ids, input_mask, segment_ids):

        fasterrcnn_features[:, -1] = crop_features
        fasterrcnn_boxes[:, -1]  = target_bbox

        # Pass the new inputs as a tuple to the LXRTEncoder
        out = self.lxrt_encoder((input_ids, input_mask, segment_ids), (fasterrcnn_features, fasterrcnn_boxes))

        if self.inputs_config['question']:
            mlp_in = out

        predictions = self.mlp(mlp_in)

        return predictions
