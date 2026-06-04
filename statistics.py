import torch, time
import torch.nn.functional as F
import torch_pruning as tp

import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def timer(model, loader, device):
    model.eval()

    time_calculation = 0.0
    amm = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            strt = time.time()
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            end = time.time()
            time_calculation += (end-strt)
            amm +=1

    print(f"Approximate time needed for forward pass of 1 batch of inputs(120) using gpu: {time_calculation/amm} seconds")



def count_parameters(model, device):
    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    print("--------------------------------------------------------")
    print(model)
    print(f"MACs: {base_macs / 1e6:.2f} M")
    print(f"Params: {base_params / 1e6:.2f} M")



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
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
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

    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names
    ))