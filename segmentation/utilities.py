from __future__ import annotations

from torch import Tensor, sum, mean, abs, maximum, max_pool2d


def dice_distance(input: Tensor, target: Tensor, spatial_dims: list[int]) -> Tensor:
    """Loss as 1 - DiceSimilarityCoefficient, averaged over non-spatial dimensions.

    :param input: some tensor with values in [0, 1]
    :param target: some tensor with values in [0, 1], with the same size as input
    :param spatial_dims: dice score is computed on these dimensions and averaged over the others; if,
        e.g. the tensors have shape (N, C, H, W), then spatial_dims should be [2, 3]
    :return: a zero-dimensional torch.Tensor
    """
    return 1 - 2 * mean(
        (sum(input * target, dim=spatial_dims) + 1e-8)
        / (sum(input + target, dim=spatial_dims) + 1e-8)
    )


def jaccard_distance(input: Tensor, target: Tensor, spatial_dims: list[int]) -> Tensor:
    """Dissimilarity of input and target classifications, averaged over non-spatial dimensions.

    :param input: some tensor with values in [0, 1]
    :param target: some tensor with values in [0, 1], with the same size as input
    :param spatial_dims: jaccard_distance is computed on these dimensions and averaged over the others; e.g. if the
        tensors have shape (N, C, H, W), then spatial_dims should be [2, 3]
    :return: a zero-dimensional torch.Tensor
    """
    return mean(
        (sum(abs(input - target), dim=spatial_dims) + 1)
        /
        (sum(maximum(input, target), dim=spatial_dims) + 1)
    )


def mj_distance(input: Tensor, target: Tensor, spatial_dims: list[int]) -> Tensor:
    perm = [i for i in range(len(input.shape)) if i not in spatial_dims] + spatial_dims
    dg_input = max_pool2d(input.permute(perm), kernel_size=4)
    dg_target = max_pool2d(target.permute(perm), kernel_size=4)
    return jaccard_distance(input, target, spatial_dims) \
           + jaccard_distance(dg_input, dg_target, [dg_input.dim() - 2, dg_input.dim() - 1])
