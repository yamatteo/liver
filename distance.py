from __future__ import annotations

from subclass_tensors import *
from torch.nn import functional


# def distance(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
#     items = {}
#     asyml1, asyml1_items = self.asyml1_df(other)
#     cross, cross_items = self.cross_entropy_df(other)
#     items.update(asyml1_items)
#     items.update(cross_items)
#     return cross, items


#
# def asyml1_df(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
#     channel_distances = torch.mean(
#         functional.relu(functional.softmax(other, dim=1) - functional.softmax(self, dim=1)),
#         dim=(2, 3, 4)
#     )
#     channel_weights = torch.tensor([[1, 5, 20]]).to(device=self.device)
#     items = {
#         "back": torch.mean(channel_distances[:, 0]).item(),
#         "livr": torch.mean(channel_distances[:, 1]).item(),
#         "tumr": torch.mean(channel_distances[:, 2]).item(),
#     }
#     return BatchDistance(torch.sum(channel_weights * channel_distances, dim=1)), items
#
def batch_cross_entropy(input: FloatSegmBatch, target: FloatSegmBatch) -> float:
    return functional.cross_entropy(input, target).item()


def individual_cross_entropy(input: FloatSegmBatch, target: FloatSegmBatch, keys: list[int]) -> dict[int, float]:
    items = {
        k: functional.cross_entropy(input.select_item(i), target.select_item(i)).item()
        for i, k in enumerate(keys)
    }
    return items


def train_cross_entropy(input: FloatSegmBatch, target: FloatSegmBatch) -> Tensor:
    return functional.cross_entropy(input, target)
