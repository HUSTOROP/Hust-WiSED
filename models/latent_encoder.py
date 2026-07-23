import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over (T, X) dimensions.
    Input:  [B, T, X, d_h]
    Output: [B, d_h]
    """
    def __init__(self, d_h: int):
        super().__init__()
        self.W_a = nn.Linear(d_h, d_h)
        self.w = nn.Linear(d_h, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h can be:
        1D: [B, T, X, d]
        2D: [B, T, X, Y, d]
        3D: [B, T, X, Y, Z, d]
        We pool over all but batch and channel dims.
        """
        if h.dim() < 3:
            raise ValueError(f"Unexpected h.dim={h.dim()}, shape={tuple(h.shape)}")

        B = h.shape[0]
        d = h.shape[-1]

        # flatten all spatiotemporal positions into one axis: N = prod(h.shape[1:-1])
        h_flat = h.reshape(B, -1, d)   # [B, N, d]
        scores = self.w(torch.tanh(self.W_a(h_flat)))  # [B, T*X, 1]
        weights = torch.softmax(scores, dim=1)  # [B, T*X, 1]
        pooled = (weights * h_flat).sum(dim=1)  # [B, d]
        return pooled


class PhysicsLatentEncoder(nn.Module):
    """
    Variational encoder mapping spatiotemporal features to structured latent z.

    Latent space decomposed into 3 semantic sub-spaces:
      z_op     : operator type preferences (dim d//3)
      z_coef   : coefficient magnitude (dim d//3)
      z_struct : equation structure complexity (dim d - 2*(d//3))

    Input:  [B, T, X, d_h]   (output of MultiResolutionEncoder1D)
    Output: z [B, d_z], mu [B, d_z], logvar [B, d_z]
    """

    def __init__(self, d_h: int = 128, d_z: int = 64):
        super().__init__()
        self.d_z = d_z
        self.d_op = d_z // 3
        self.d_coef = d_z // 3
        self.d_struct = d_z - 2 * (d_z // 3)

        # Attention pooling
        self.attn_pool = AttentionPooling(d_h)

        # MLP encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h // 2),
            nn.GELU(),
        )

        # Variational heads
        self.fc_mu = nn.Linear(d_h // 2, d_z)
        self.fc_logvar = nn.Linear(d_h // 2, d_z)

        # Semantic sub-space projection heads (for structured prior)
        # These project each semantic sub-space for loss computation
        self.proj_op = nn.Linear(self.d_op, self.d_op)
        self.proj_coef = nn.Linear(self.d_coef, self.d_coef)
        self.proj_struct = nn.Linear(self.d_struct, self.d_struct)

    def encode(self, h: torch.Tensor):
        """
        h: [B, T, X, d_h]
        Returns: mu [B, d_z], logvar [B, d_z]
        """
        # Global pooling
        pooled = self.attn_pool(h)  # [B, d_h]

        # MLP
        hidden = self.encoder_mlp(pooled)  # [B, d_h//2]

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        logvar = torch.clamp(logvar, -10, 2)  # numerical stability

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        During inference (eval mode), returns mu directly.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(self, h: torch.Tensor):
        """
        h: [B, T, X, d_h]
        Returns: z [B, d_z], mu [B, d_z], logvar [B, d_z]
        """
        mu, logvar = self.encode(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence: D_KL(N(mu, sigma^2) || N(0, I))
        = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def struct_sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        L1 sparsity on the structure sub-space z_struct.
        Encourages discovering simpler equations.
        """
        z_struct = z[:, self.d_op + self.d_coef:]
        return torch.mean(torch.abs(z_struct))

    def get_subspaces(self, z: torch.Tensor):
        """Split z into semantic sub-spaces."""
        z_op = z[:, :self.d_op]
        z_coef = z[:, self.d_op:self.d_op + self.d_coef]
        z_struct = z[:, self.d_op + self.d_coef:]
        return z_op, z_coef, z_struct


if __name__ == "__main__":
    enc = PhysicsLatentEncoder(d_h=128, d_z=64)
    h = torch.randn(4, 101, 64, 128)  # [B, T, X, d_h]
    z, mu, logvar = enc(h)
    print(f"PLE: h {h.shape} -> z {z.shape}, mu {mu.shape}")
    print(f"KL loss: {enc.kl_loss(mu, logvar).item():.4f}")
    params = sum(p.numel() for p in enc.parameters())
    print(f"Parameters: {params:,}")

