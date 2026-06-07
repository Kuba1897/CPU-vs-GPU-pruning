import torch.nn.utils.prune as prune
import torch.nn as nn
import torch
import torch_pruning as tp

def apply_unstructured_pruning(model, part_step:bool=False, amount=0.5):
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))


    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    if part_step:
        return

    total = 0
    nonzero = 0
    trainable = 0

    mask_removement(model)

    for p in model.parameters():
        total += p.numel()
        nonzero += torch.count_nonzero(p).item()
        if p.requires_grad:
            trainable += p.numel()

    sparsity = 1 - nonzero / total
    print(f"Sparsity: {sparsity}")
    print(f"Zeros: {total-nonzero}")
    print(f"Total: {total}")

        

def mask_removement(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")
        if isinstance(module, nn.Linear):
            prune.remove(module, "weight")


def apply_structured_pruning(model,device, regularization, amount=0.5):
    model.eval()

    example_inputs = torch.randn(1, 3, 32, 32).to(device)

    # L2-norm importance dla grup kanałów
    importance = tp.importance.GroupMagnitudeImportance(p=regularization)

    # Nie przycinaj ostatniej warstwy klasyfikującej.
    # Możesz też ignorować pierwszą Conv2d, jeśli accuracy mocno spada.
    ignored_layers = []

    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    pruner = tp.pruner.BasePruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=amount,  
        ignored_layers=ignored_layers,
        round_to=8,             # zaokrąglenie liczby kanałów do wielokrotności 8
    )

    # Faktyczne zmniejszenie modelu
    pruner.step()