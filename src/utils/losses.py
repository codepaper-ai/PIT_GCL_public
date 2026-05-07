import torch
import torch.nn.functional as F


def nt_xent_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss for protein-protein interaction pairs.

    Positive pairs: (z_a[i], z_b[i]) where labels[i] == 1.
    All other (z_a[i], z_b[j]) pairs with j != i are treated as negatives.

    Args:
        z_a: Embeddings for protein A, shape (B, d)
        z_b: Embeddings for protein B, shape (B, d)
        labels: Binary labels, shape (B,)
        temperature: Softmax temperature

    Returns:
        Scalar contrastive loss
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    sim = torch.mm(z_a, z_b.t()) / temperature  # (B, B)

    pos_mask = labels.bool()
    if pos_mask.sum() == 0:
        return z_a.new_zeros(1).squeeze()

    idx = torch.arange(sim.size(0), device=z_a.device)
    loss_a2b = F.cross_entropy(sim[pos_mask], idx[pos_mask])
    loss_b2a = F.cross_entropy(sim.t()[pos_mask], idx[pos_mask])
    return (loss_a2b + loss_b2a) / 2.0


def ppi_loss(
    y_pred: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    temperature: float = 0.07,
) -> tuple:
    """
    Multi-objective loss: L_total = alpha * L_contrastive + beta * L_BCE.

    Args:
        y_pred: Predicted binding probabilities, shape (B,) or (B, 1)
        z_a: Pooled embedding for protein A, shape (B, d)
        z_b: Pooled embedding for protein B, shape (B, d)
        labels: Binary labels, shape (B,)
        alpha: Weight for contrastive loss
        beta: Weight for BCE loss
        temperature: NT-Xent temperature

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual components
    """
    l_bce = F.binary_cross_entropy(y_pred.view(-1), labels.float())
    l_cont = nt_xent_loss(z_a, z_b, labels, temperature)
    total = alpha * l_cont + beta * l_bce
    return total, {
        "bce": l_bce.item(),
        "contrastive": l_cont.item(),
        "total": total.item(),
    }
