import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

AVAILABLE_EXTRACTORS = ['mcait', 'resnet18', 'wide_resnet50_2']


def get_feature_extractor(backbone, input_size=256, **kwargs):
    assert backbone in AVAILABLE_EXTRACTORS, f"Feature extractor must be one of {AVAILABLE_EXTRACTORS}."
    if backbone in ["resnet18", "wide_resnet50_2"]:
        return ResNetFeatureExtractor(backbone, input_size, **kwargs)
    elif backbone == "mcait":
        return MCaitFeatureExtractor(**kwargs)
    raise ValueError("`backbone` must be one of `[mcait, resnet18, wide_resnet50_2]`")


class FeatureExtractorInterface(L.LightningModule):

    def __init__(self):
        super(FeatureExtractorInterface, self).__init__()

    def forward(self, img, **kwargs):
        raise NotImplementedError

    def extract_features(self, img, **kwargs):
        raise NotImplementedError

    def normalize_features(self, x, **kwargs):
        raise NotImplementedError


class ResNetFeatureExtractor(FeatureExtractorInterface):
    def __init__(self, backbone, input_size, max_downsampling_factor=32, **kwargs):
        super(ResNetFeatureExtractor, self).__init__()
        self.extractor = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
            output_stride=max_downsampling_factor
        )
        self.channels = self.extractor.feature_info.channels()
        self.scale_factors = self.extractor.feature_info.reduction()
        self.scales = range(len(self.scale_factors))

        self.feature_normalizations = nn.ModuleList()
        for in_channels, scale in zip(self.channels, self.scale_factors):
            self.feature_normalizations.append(nn.LayerNorm(
                [in_channels, int(input_size / scale), int(input_size / scale)],
                elementwise_affine=True
            ))

        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, img, **kwargs):
        features = self.extract_features(img)
        normalized_features = self.normalize_features(features, **kwargs)
        return normalized_features

    def extract_features(self, img, **kwargs):
        self.extractor.eval()
        return self.extractor(img)

    def normalize_features(self, features, **kwargs):
        return [self.feature_normalizations[i](feature) for i, feature in enumerate(features)]


class MCaitFeatureExtractor(FeatureExtractorInterface):
    def __init__(self, **kwargs):
        super(MCaitFeatureExtractor, self).__init__()
        self.input_size = 448
        self.extractor1 = timm.create_model("cait_m48_448", pretrained=True)
        self.extractor2 = timm.create_model("cait_s24_224", pretrained=True)
        self.channels = [768, 384]
        self.scale_factors = [16, 32]
        self.scales = range(len(self.scale_factors))

        for param in self.extractor1.parameters():
            param.requires_grad = False
        for param in self.extractor2.parameters():
            param.requires_grad = False

    def forward(self, img, training=True):
        features = self.extract_features(img)
        normalized_features = self.normalize_features(features, training=training)
        return normalized_features

    def extract_features(self, img, **kwargs):
        self.extractor1.eval()
        self.extractor2.eval()

        # MPS WALK-AROUND
        original_device = img.device.type
        tmp_device = "cpu"
        if original_device == 'mps':
            img = img.to(tmp_device)
            self.extractor1 = self.extractor1.to(tmp_device)
            self.extractor2 = self.extractor2.to(tmp_device)

        # Scale 1 --> Extractor 1
        x1 = self.extractor1.patch_embed(img)
        x1 = x1 + self.extractor1.pos_embed
        x1 = self.extractor1.pos_drop(x1)
        for i in range(41):  # paper Table 6. Block Index = 40
            x1 = self.extractor1.blocks[i](x1)

        # Scale 2 --> Extractor 2
        img_sub = F.interpolate(torch.Tensor(img), size=(224, 224), mode='bicubic', align_corners=True)
        x2 = self.extractor2.patch_embed(img_sub)
        x2 = x2 + self.extractor2.pos_embed
        x2 = self.extractor2.pos_drop(x2)
        for i in range(21):
            x2 = self.extractor2.blocks[i](x2)

        features = [x1, x2]

        # MPS WALK-AROUND
        if original_device == 'mps':
            features = [f.to(original_device) for f in features]
            self.extractor1 = self.extractor1.to(original_device)
            self.extractor2 = self.extractor2.to(original_device)

        return features

    def normalize_features(self, features, **kwargs):

        normalized_features = []
        for i, extractor in enumerate([self.extractor1, self.extractor2]):
            batch, _, channels = features[i].shape
            scale_factor = self.scale_factors[i]

            x = extractor.norm(features[i].contiguous())
            x = x.permute(0, 2, 1)
            x = x.reshape(batch, channels, self.input_size // scale_factor, self.input_size // scale_factor)
            normalized_features.append(x)

        return normalized_features
