import torch
import time
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    time_calculation = 0.0
    amm = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            start = time.time()

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            end = time.time()

            time_calculation += (end - start)
            amm += 1

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total, time_calculation / amm


def count_parameters(model):
    total = 0
    nonzero = 0
    trainable = 0

    visited = set()

    # obsługa modeli z aktywnym pruningiem
    for module in model.modules():

        if hasattr(module, "weight_mask") and hasattr(module, "weight_orig"):

            effective_weight = module.weight_orig * module.weight_mask

            total += effective_weight.numel()
            nonzero += torch.count_nonzero(effective_weight).item()

            if module.weight_orig.requires_grad:
                trainable += effective_weight.numel()

            visited.add(id(module.weight_orig))

            if getattr(module, "bias", None) is not None:
                bias = module.bias

                total += bias.numel()
                nonzero += torch.count_nonzero(bias).item()

                if bias.requires_grad:
                    trainable += bias.numel()

                visited.add(id(bias))

    # pozostałe parametry
    for param in model.parameters():

        if id(param) in visited:
            continue

        total += param.numel()
        nonzero += torch.count_nonzero(param).item()

        if param.requires_grad:
            trainable += param.numel()

    sparsity = 1.0 - (nonzero / total)

    return total, trainable, nonzero, sparsity


def model_size_mb(model):
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / (1024 ** 2)


def collect_predictions(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs, all_labels


def roc_and_statistics(model, dataset, device, class_names):
    y_score, y_true = collect_predictions(model, dataset, device)

    num_classes = 10

    y_true_bin = label_binarize(
        y_true,
        classes=list(range(num_classes))
    )

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_score[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[i]} AUC={roc_auc:.3f}"
        )

    plt.plot([0, 1], [0, 1], "k--", label="random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves for CIFAR-10 classes")
    plt.legend()
    plt.grid()
    plt.show()

    y_pred = y_score.argmax(axis=1)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )