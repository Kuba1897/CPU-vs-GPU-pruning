import torch
import torch.nn as nn
from models import PrunableCNN


def get_kept_out_channels(conv: nn.Conv2d):
    """
    Zwraca indeksy kanałów wyjściowych, które nie zostały całkowicie wyzerowane.
    Działa po structured pruningu dim=0.
    """
    if hasattr(conv, "weight_mask"):
        mask = conv.weight_mask.detach()
        kept = mask.view(mask.shape[0], -1).sum(dim=1) != 0
    else:
        weight = conv.weight.detach()
        kept = weight.view(weight.shape[0], -1).abs().sum(dim=1) != 0

    return torch.where(kept)[0]


def shrink_conv2d(old_conv, kept_in=None, kept_out=None):
    """
    Tworzy nową Conv2d z wybranymi kanałami wejściowymi i wyjściowymi.

    kept_in  - indeksy kanałów wejściowych, które zostają
    kept_out - indeksy filtrów wyjściowych, które zostają
    """

    if kept_in is None:
        kept_in = torch.arange(old_conv.in_channels)

    if kept_out is None:
        kept_out = torch.arange(old_conv.out_channels)

    new_conv = nn.Conv2d(
        in_channels=len(kept_in),
        out_channels=len(kept_out),
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        weight = old_conv.weight.detach()

        # [out_channels, in_channels, kH, kW]
        new_conv.weight.copy_(weight[kept_out][:, kept_in, :, :])

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.detach()[kept_out])

    return new_conv


def shrink_batchnorm2d(old_bn, kept):
    new_bn = nn.BatchNorm2d(
        num_features=len(kept),
        eps=old_bn.eps,
        momentum=old_bn.momentum,
        affine=old_bn.affine,
        track_running_stats=old_bn.track_running_stats,
    )

    with torch.no_grad():
        if old_bn.affine:
            new_bn.weight.copy_(old_bn.weight.detach()[kept])
            new_bn.bias.copy_(old_bn.bias.detach()[kept])

        if old_bn.track_running_stats:
            new_bn.running_mean.copy_(old_bn.running_mean.detach()[kept])
            new_bn.running_var.copy_(old_bn.running_var.detach()[kept])
            new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)

    return new_bn


def rebuild_pruned_features(old_model):
    old_features = old_model.features

    conv_indices = [0, 3, 7, 10, 14, 17]
    bn_indices = [1, 4, 8, 11, 15, 18]

    # indeksy kanałów wyjściowych, które zostały po pruningu każdej Conv2d
    kept_out = {}

    for idx in conv_indices:
        kept_out[idx] = get_kept_out_channels(old_features[idx]).cpu()

    new_layers = []

    previous_kept = None

    for conv_idx, bn_idx in zip(conv_indices, bn_indices):
        old_conv = old_features[conv_idx]
        old_bn = old_features[bn_idx]

        current_kept = kept_out[conv_idx]

        # dla pierwszej Conv2d wejściem jest RGB, czyli 3 kanały
        if previous_kept is None:
            kept_in = torch.arange(old_conv.in_channels)
        else:
            kept_in = previous_kept

        new_conv = shrink_conv2d(
            old_conv,
            kept_in=kept_in,
            kept_out=current_kept
        )

        new_bn = shrink_batchnorm2d(
            old_bn,
            kept=current_kept
        )

        new_layers.append(new_conv)
        new_layers.append(new_bn)
        new_layers.append(nn.ReLU())

        # po drugiej, czwartej i szóstej konwolucji masz MaxPool
        if conv_idx in [3, 10, 17]:
            new_layers.append(nn.MaxPool2d(2))

        previous_kept = current_kept

    return nn.Sequential(*new_layers), kept_out[17]



def rebuild_classifier(old_model, last_kept_channels, num_classes=10):
    old_classifier = old_model.classifier

    old_linear1 = old_classifier[1]
    old_linear2 = old_classifier[4]

    new_in_features = len(last_kept_channels) * 4 * 4

    new_linear1 = nn.Linear(new_in_features, old_linear1.out_features)
    new_linear2 = nn.Linear(old_linear2.in_features, old_linear2.out_features)

    # Musimy przepisać tylko wejścia odpowiadające zachowanym kanałom z ostatniej Conv2d.
    # Stary Flatten robi układ: [C, H, W], czyli kanały po 4*4 wartości.
    kept_flat_indices = []

    for c in last_kept_channels:
        start = int(c) * 4 * 4
        end = start + 4 * 4
        kept_flat_indices.extend(range(start, end))

    kept_flat_indices = torch.tensor(kept_flat_indices)

    with torch.no_grad():
        new_linear1.weight.copy_(old_linear1.weight.detach()[:, kept_flat_indices])
        new_linear1.bias.copy_(old_linear1.bias.detach())

        new_linear2.weight.copy_(old_linear2.weight.detach())
        new_linear2.bias.copy_(old_linear2.bias.detach())

    new_classifier = nn.Sequential(
        nn.Flatten(),
        new_linear1,
        nn.ReLU(),
        nn.Dropout(0.4),
        new_linear2
    )

    return new_classifier



def rebuild_pruned_model(old_model, num_classes=10):
    new_model = PrunableCNN(num_classes=num_classes)

    new_features, last_kept_channels = rebuild_pruned_features(old_model)
    new_classifier = rebuild_classifier(
        old_model,
        last_kept_channels=last_kept_channels,
        num_classes=num_classes
    )

    new_model.features = new_features
    new_model.classifier = new_classifier

    return new_model