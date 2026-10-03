import torch


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure input has shape [B, T, D]."""
    if x.dim() == 2:  # [T, D] -> [1, T, D]
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError("trajectory must be [B, T, D] or [T, D]")
    return x


def acceleration_smoothness(
    trajectory: torch.Tensor,
    action_dim: int = 2,
    dt: float = 1e-2,
    eps_dt: float = 1e-12,
) -> torch.Tensor:
    """
    Acceleration smoothness (Sm): mean over interior states of || p_{k+1} - 2 p_k + p_{k-1} || / dt^2,
    with p = (x, y) = trajectory[..., [action_dim + 1, action_dim]].
    Returns one value per batch element; a scalar if the input was [T, D].
    """
    X = _ensure_batch(trajectory)
    B, T, _ = X.shape
    if T < 3:
        zeros = X.new_zeros((B,))
        return zeros[0] if trajectory.dim() == 2 else zeros

    y = X[:, :, action_dim]
    x = X[:, :, action_dim + 1]
    P = torch.stack([x, y], dim=-1)

    second_diff = P[:, 2:, :] - 2.0 * P[:, 1:-1, :] + P[:, :-2, :]

    dt2 = torch.as_tensor(dt, dtype=P.dtype, device=P.device)
    dt2 = (dt2 * dt2).clamp(min=eps_dt * eps_dt)

    a_vec = second_diff / dt2
    a_mag = torch.linalg.norm(a_vec, dim=-1)

    mask = torch.isfinite(a_mag)
    sums = torch.where(mask, a_mag, torch.zeros_like(a_mag)).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    s_smooth = sums / counts

    return s_smooth[0] if trajectory.dim() == 2 else s_smooth
