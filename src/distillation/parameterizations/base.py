import torch
from torch import nn


def _kmeans(data, k, n_iter=100):
    """Lloyd's K-means on arbitrary tensors. Returns [k, D] cluster centers."""
    num_points = data.shape[0]
    indices = torch.randperm(num_points, device=data.device)[:k]
    centers = data[indices].clone().float()
    data_f = data.float()
    for _ in range(n_iter):
        dists = torch.cdist(data_f, centers)
        labels = dists.argmin(dim=-1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=data.device)
        new_centers.index_add_(0, labels, data_f)
        counts.index_add_(0, labels, torch.ones(num_points, device=data.device))
        mask = counts > 0
        new_centers[mask] /= counts[mask].unsqueeze(-1)
        empty = (~mask).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            new_centers[empty] = data_f[
                torch.randint(num_points, (empty.numel(),), device=data.device)
            ]
        centers = new_centers
    return centers.to(data.dtype)


class BaseSyntheticTokenDataset(nn.Module):
    def sample_indices(self, batch_size, device):
        if batch_size >= self.num_sequences:
            return torch.arange(self.num_sequences, device=device)
        return torch.randperm(self.num_sequences, device=device)[:batch_size]

    def token_probs(self, indices=None, embedding_weight=None):
        raise NotImplementedError

    def input_embeds(self, indices, embedding_weight):
        return self.token_probs(indices, embedding_weight) @ embedding_weight

    def forward(self, indices=None, embedding_weight=None):
        return self.token_probs(indices, embedding_weight)

    @torch.no_grad()
    def hard_tokens(self, embedding_weight=None):
        return self.token_probs(embedding_weight=embedding_weight).argmax(dim=-1)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
