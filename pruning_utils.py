import torch.nn.utils.prune as prune
import torch.nn as nn

def apply_unstructured_pruning(model, amount=0.5):
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    for module, name in parameters_to_prune:
        prune.remove(module, name)

def apply_structured_pruning(model, regularization, amount=0.5): #regularization może być 1 albo 2
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module=module,
                name="weight",
                amount=amount,
                n=regularization,
                dim=0
            )
        if isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=regularization, dim=1)
        
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, "weight")
        if isinstance(module, nn.Linear):
            prune.remove(module, "weight")