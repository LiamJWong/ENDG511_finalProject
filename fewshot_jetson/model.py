import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def get_convnet(output_size, pretrained=False):
    """
    Build the same MobileNetV3-Small backbone used in training,
    replacing the final classifier layer with an embedding output.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, output_size)
    return model


class ProtoNet(nn.Module):
    def __init__(self, proto_dim=64, lr=None):
        """
        lr is kept as an optional argument only for compatibility with
        notebook-style initialization, but it is not used in deployment.
        """
        super().__init__()
        self.proto_dim = proto_dim
        self.lr = lr
        self.model = get_convnet(output_size=proto_dim, pretrained=False)

    def encode(self, imgs):
        return self.model(imgs)

    @staticmethod
    def calculate_prototypes(features, targets):
        """
        features: [N, D]
        targets:  [N]
        returns:
            prototypes: [num_classes, D]
            classes:    sorted class ids
        """
        classes, _ = torch.unique(targets).sort()
        prototypes = []

        for c in classes:
            idx = torch.where(targets == c)[0]
            proto = features[idx].mean(dim=0)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, classes

    @staticmethod
    def classify_feats(prototypes, classes, feats):
        """
        prototypes: [C, D]
        feats:      [B, D]
        returns log-prob-like scores and class ids
        """
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)
        preds = F.log_softmax(-dist, dim=1)
        return preds, classes

    def predict(self, support_imgs, support_targets, query_imgs):
        """
        End-to-end few-shot prediction helper for deployment.
        """
        support_feats = self.encode(support_imgs)
        query_feats = self.encode(query_imgs)

        prototypes, classes = self.calculate_prototypes(support_feats, support_targets)
        preds, classes = self.classify_feats(prototypes, classes, query_feats)
        return preds, classes