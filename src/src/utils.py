import torch

def compute_sparsity(model, threshold=1e-2):
    total, zero = 0, 0
    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()
    return 100 * zero / total
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            loss += gates.sum()
    return loss
