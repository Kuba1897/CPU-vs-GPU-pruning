import torch
import torch.nn.utils.prune as prune

def apply_unstructured_pruning(model, amount=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def apply_structured_pruning(model, amount=0.5):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=1)
    return model