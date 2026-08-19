import torch
import torch.nn as nn


class KNN(nn.Module):
    def __init__(self, k=16, transpose_mode=True):
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, ref, query):
        if not self.transpose_mode:
            # [B, D, N] -> [B, N, D]
            ref = ref.transpose(1, 2)
            query = query.transpose(1, 2)

        dist = torch.cdist(query, ref)
        dist, idx = torch.topk(
            dist,
            k=self.k,
            dim=-1,
            largest=False,
            sorted=True,
        )

        if not self.transpose_mode:
            # [B, Nq, K] -> [B, K, Nq]
            dist = dist.transpose(1, 2)
            idx = idx.transpose(1, 2)

        return dist, idx