import collections
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from FrEIA import framework as ff, modules as fm

from src.feature_extraction import get_feature_extractor


class UFlow(L.LightningModule):
    def __init__(self, input_size=256, flow_steps=8, backbone="mcait"):
        super(UFlow, self).__init__()
        self.input_size = input_size

        # Feature Extractor
        self.feature_extractor = get_feature_extractor(backbone, input_size)

        # Flow Graph
        input_nodes = []
        for channel, s_factor in zip(self.feature_extractor.channels, self.feature_extractor.scale_factors):
            input_nodes.append(
                ff.InputNode(channel, input_size // s_factor, input_size // s_factor, name=f"cond_{channel}")
            )

        nodes, output_nodes = [], []
        last_node = input_nodes[-1]
        for i in reversed(range(1, len(input_nodes))):
            flows = self.get_flow_stage(last_node, flow_steps)
            volume_size = flows[-1].output_dims[0][0]
            split = ff.Node(
                flows[-1], fm.Split,
                {'section_sizes': (volume_size // 8 * 4, volume_size - volume_size // 8 * 4), 'dim': 0},
                name=f'split_{i + 1}'
            )
            output = ff.OutputNode(split.out1, name=f'output_scale_{i + 1}')
            up = ff.Node(split.out0, fm.IRevNetUpsampling, {}, name=f'up_{i + 1}')
            last_node = ff.Node([input_nodes[i - 1].out0, up.out0], fm.Concat, {'dim': 0}, name=f'cat_{i}')

            output_nodes.append(output)
            nodes.extend([*flows, split, up, last_node])

        flows = self.get_flow_stage(last_node, flow_steps)
        output = ff.OutputNode(flows[-1], name='output_scale_1')

        output_nodes.append(output)
        nodes.extend(flows)

        self.flow = ff.GraphINN(input_nodes + nodes + output_nodes[::-1])

    @staticmethod
    def get_flow_stage(in_node, flow_steps, condition_node=None):

        def get_affine_coupling_subnet(kernel_size, subnet_channels_ratio):
            def affine_coupling_subnet(in_channels, out_channels):
                mid_channels = int(in_channels * subnet_channels_ratio)
                return nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size, padding="same"),
                    nn.ReLU(),
                    nn.Conv2d(mid_channels, out_channels, kernel_size, padding="same"),
                )

            return affine_coupling_subnet

        def glow_params(k):
            return {
                'subnet_constructor': get_affine_coupling_subnet(k, 1.0),
                'affine_clamping': 2.0,
                'permute_soft': False
            }

        flow_size = in_node.output_dims[0][-1]
        nodes = []
        for step in range(flow_steps):
            nodes.append(ff.Node(
                in_node,
                fm.AllInOneBlock,
                glow_params(3 if step % 2 == 0 else 1),
                conditions=condition_node,
                name=f"flow{flow_size}_step{step}"
            ))
            in_node = nodes[-1]
        return nodes

    def forward(self, image):
        features = self.feature_extractor(image)
        return self.encode(features)

    def encode(self, features):
        z, ljd = self.flow(features, rev=False)
        if len(self.feature_extractor.scales) == 1:
            z = [z]
        return z, ljd

    def get_probability(self, outputs, resize_size=None):
        if resize_size is None:
            resize_size = self.input_size

        probabilities = []
        for output in outputs:
            log_prob_i = -torch.mean(output ** 2, dim=1, keepdim=True) * 0.5
            prob_i = torch.exp(log_prob_i)
            probabilities.append(F.interpolate(
                prob_i,
                size=[resize_size, resize_size],
                mode="bilinear",
                align_corners=False,
            ))
        return torch.mean(torch.stack(probabilities, dim=-1), dim=-1)

    def from_pretrained(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"]
        state_dict = collections.OrderedDict({k.split("model.")[-1]: v for k, v in state_dict.items()})

        for k in [k for k in state_dict.keys() if 'log_loss_weights' in k]:
            del state_dict[k]
        for k in [k for k in state_dict.keys() if 'aupro' in k]:
            del state_dict[k]
        for k in [k for k in state_dict.keys() if 'f1' in k]:
            del state_dict[k]
        for k in [k for k in state_dict.keys() if 'mcc' in k]:
            del state_dict[k]

        self.load_state_dict(state_dict)
