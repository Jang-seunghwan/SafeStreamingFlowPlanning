import torch
import math


class GuidanceMatcher:
    """
    Cov-G reward guidance for flow matching:
        v_guided(x, t) = v(x, t) + scale * s(t) * grad V(x1_pred),  s(t) = (1 + cos(pi t)) / 2
    """
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def schedule_fn(self, t):
        return 0.5 * (1 + torch.cos(t * math.pi))

    def apply_guidance(self, vt, grad_v, t):
        """
        Args:
            vt: vector field predicted by the model (B, horizon, transition_dim)
            grad_v: gradient of the value function at the predicted endpoint
            t: current flow time (B,)
        """
        return vt + grad_v * self.scale * self.schedule_fn(t)
