"""WP backbone extracted from the author source; see provenance.json."""
import math
import torch
from torch import nn
from torch.nn import Linear
from torch.nn import functional as F

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps.
  Basically it multiplexes a scalar `t` into a vector of `sin(2 pi k t)` and `cos(2 pi k t)` features.
  """
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, t):
    t_proj = t.view(-1, 1) * self.W[None, :] * 2 * math.pi
    return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)



class UNetMLPBlock(torch.nn.Module):
    def __init__(self,
        in_features, out_features, emb_features, dropout=0, skip_scale=1, eps=1e-5,
        adaptive_scale=True, init=dict(), init_zero=dict(),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.emb_features = emb_features
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = nn.LayerNorm(in_features, eps=eps) #GroupNorm(num_channels=in_features, eps=eps)
        self.fc0 = Linear(in_features=in_features, out_features=out_features, **init)
        self.affine = Linear(in_features=emb_features, out_features=out_features*(2 if adaptive_scale else 1), **init)
        self.norm1 = nn.LayerNorm(out_features, eps=eps) #GroupNorm(num_channels=out_features, eps=eps)
        self.fc1 = Linear(in_features=out_features, out_features=out_features, **init_zero)

        self.skip = None
        if out_features != in_features:
            self.skip = Linear(in_features=in_features, out_features=out_features, **init)

    def forward(self, x, emb):
        orig = x
        x = self.fc0(F.silu(self.norm0(x)))

        params = self.affine(emb).to(x.dtype) # .unsqueeze(1)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))

        x = self.fc1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        return x
    
    

class UNetBlockStyleMLP_backbone(nn.Module):
  """A time-dependent score-based model."""

  def __init__(self, ndim=2, nlayers=5, nhidden=64, time_embed_dim=64,):
    super().__init__()
    self.embed = GaussianFourierProjection(time_embed_dim, scale=1)
    layers = nn.ModuleList()
    layers.append(UNetMLPBlock(ndim, nhidden, time_embed_dim))
    for _ in range(nlayers-2):
        layers.append(UNetMLPBlock(nhidden, nhidden, time_embed_dim))
    layers.append(nn.Linear(nhidden, ndim))
    self.net = layers

  def forward(self, x, t_enc, cond=None):
    # t_enc : preconditioned version of sigma, usually 
    # ln_std_vec = torch.log(std_vec) / 4
    t_embed = self.embed(t_enc)
    for layer in self.net[:-1]:
        x = layer(x, t_embed)
    pred = self.net[-1](x)
    return pred



class Denoiser(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.model = UNetBlockStyleMLP_backbone(d, 5, 256, 256)
        self.sigma_data = 0.5
    def forward(self, y, sigma):
        sigma = sigma.reshape(-1, 1)
        cskip = 0.25 / (sigma.square() + 0.25)
        cout = sigma * 0.5 / (sigma.square() + 0.25).sqrt()
        cin = (sigma.square() + 0.25).rsqrt()
        return cskip*y + cout*self.model(cin*y, sigma.log()/4)

