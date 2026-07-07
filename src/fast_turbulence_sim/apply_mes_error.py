import torch
import torch.nn as nn
import numpy  as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════════

class MixtureDensityNetwork(nn.Module):
    """
    MDN: X → MLP → (π, μ, σ) for K Gaussian components.

    Args:
        input_dim:    Dimensionality of X.
        hidden_dim:   Width of each hidden layer.
        n_components: Number of Gaussian mixture components K.
                      10–20 is usually enough; increase for very complex shapes.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 n_components: int = 10):
        super().__init__()
        self.K = n_components

        # Shared trunk
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        # Output heads — one value per mixture component
        self.pi_head      = nn.Linear(hidden_dim, n_components)  # mixing weights
        self.mu_head      = nn.Linear(hidden_dim, n_components)  # Gaussian means
        self.log_sig_head = nn.Linear(hidden_dim, n_components)  # log std devs

        # Normalisation stats — stored as buffers so they're saved with state_dict
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std",  torch.ones(input_dim))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std",  torch.ones(1))

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """Returns (π, μ, σ), each of shape (batch, K)."""
        h     = self.net(x)
        pi    = torch.softmax(self.pi_head(h), dim=-1)                     # (B, K)
        mu    = self.mu_head(h)                                            # (B, K)
        sigma = torch.exp(self.log_sig_head(h)).clamp(min=1e-6, max=1e3)  # (B, K)
        return pi, mu, sigma

    # ── loss ──────────────────────────────────────────────────────────────────
    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the Gaussian mixture (minimise this)."""
        pi, mu, sigma = self(x)
        # Log-prob of y under each component: (B, K)
        log_p = torch.distributions.Normal(mu, sigma).log_prob(y.unsqueeze(-1))
        # Numerically stable log-sum-exp over components
        log_mix = torch.logsumexp(torch.log(pi + 1e-8) + log_p, dim=-1)   # (B,)
        return -log_mix.mean()

    # ── sampling ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Draw one Y sample per row of x (in normalised space).
        Fully vectorised — efficient on both CPU and GPU.
        """
        pi, mu, sigma = self(x)
        k       = torch.multinomial(pi, 1)         # (B, 1)  component index
        mu_k    = mu.gather(1, k).squeeze(1)       # (B,)    selected mean
        sigma_k = sigma.gather(1, k).squeeze(1)    # (B,)    selected std dev
        return torch.normal(mu_k, sigma_k)         # (B,)

    # ── normalisation helpers ─────────────────────────────────────────────────
    def _norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def _denorm_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(
    model:     MixtureDensityNetwork,
    X_new:     np.ndarray,           # (N, D) or (N,)
    n_samples: int = 1,
) -> np.ndarray:
    """
    For each row of X_new, draw n_samples from P(Y | X).

    Returns ndarray of shape (N, n_samples).
    The call is fully vectorised: all N·n_samples points are processed
    in a single forward pass, making it fast even for large batches.
    """
    model.eval()
    device = next(model.parameters()).device
    X_new = np.atleast_2d(X_new.T).T.astype(np.float32)
    N     = len(X_new)

    Xt = torch.from_numpy(X_new).to(device)
    Xn = model._norm_x(Xt)                               # normalise

    # Tile rows: (N, D) → (N·S, D) so each input gets S independent draws
    Xn_rep  = Xn.repeat_interleave(n_samples, dim=0)    # (N·S, D)
    samples = model.sample(Xn_rep)                       # (N·S,)
    samples = model._denorm_y(samples)                   # de-normalise

    return samples.cpu().numpy().reshape(N, n_samples)


class MesError(object):
    def __init__(self,
                 model_path: str,
                 binning: object):
        self.model = MixtureDensityNetwork(input_dim=2,
                                           hidden_dim=128,
                                           n_components=10)
        self.model.load_state_dict(torch.load(model_path,
                                              map_location="cpu")
                                   )
        self.model.eval()

        half_map_shape = binning.shape[0]//2

        self.radii = np.sqrt(  (binning.xBar_bins - half_map_shape) ** 2
                             + (binning.yBar_bins - half_map_shape) ** 2)

    def apply(self,
              vec_input: np.ndarray):

        if len(vec_input.shape) == 1:
            vec_input = vec_input.reshape(1, -1)

        radii = np.repeat(self.radii[None,:], vec_input.shape[0], axis=0)

        stacked_array = np.vstack([vec_input.flatten(),
                                   radii.flatten()]
                                  ).T


        prediction = predict(self.model, stacked_array, n_samples=1)

        return prediction.reshape(vec_input.shape[0],
                                  self.radii.shape[0])