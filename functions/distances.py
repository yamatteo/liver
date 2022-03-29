from __future__ import annotations

from torch import Tensor, sum, mean, abs, maximum, max_pool2d, sqrt


def dice_distance(input: Tensor, target: Tensor, spatial_dims: list[int]) -> Tensor:
    """Loss as 1 - DiceSimilarityCoefficient, averaged over non-spatial dimensions.

    :param input: some tensor (N, C, H, W, D) with values in [0, 1]
    :param target: some tensor (N, C, H, W, D) with values in [0, 1], with the same size as input
    :return: a zero-dimensional torch.Tensor
    """
    return 1 - 2 * mean(
        sum(input * target, dim=(2, 3, 4))
        / (sum(input + target, dim=(2, 3, 4)) + 1)
    )


def jaccard_distance(input: Tensor, target: Tensor) -> Tensor:
    """Dissimilarity of input and target classifications.

    :param input: some tensor (N, C, H, W, D) with values in [0, 1]
    :param target: some tensor (N, C, H, W, D) with values in [0, 1], with the same size as input
    :return: a zero dimensional tensor
    """
    return mean(
        sum(abs(input - target), dim=(2, 3, 4))
        / (sum(maximum(input, target), dim=(2, 3, 4)) + 1),
    )


# def halfway_jaccard_distance(input: Tensor, target: Tensor) -> Tensor:
#     """Dissimilarity of input and target classifications.
#
#     :param input: some tensor (N, C, H, W, D) with values in [0, 1]
#     :param target: some tensor (N, C, H, W, D) with values in [0, 1], with the same size as input
#     :return: a tensor (N, )
#     """
#     return sum(
#         sum(abs(input - target), dim=(2, 3, 4))
#         / sqrt(sum(maximum(input, target), dim=(2, 3, 4)) + 1),
#         dim=1
#     )