from torch import Tensor, quantile, linspace, clamp, mean


def quantiles_pxyz2qz(input: Tensor) -> Tensor:
    """From scan (Phases, X, Y, Z) calculates 10 quantiles (Z, Quantiles)."""
    phases, x, y, z = input.shape
    return quantile(
        input=input.reshape(phases, x * y, z).float(),
        q=linspace(0.1, 1, 10),
        dim=1,
    ).permute(2, 1, 0).reshape(z, 10 * phases)


def relevance_xyz2z(input: Tensor) -> Tensor:
    """From segmentation (X, Y, Z) calculates relevance (Z, )."""
    x, y, z = input.shape
    return mean(clamp(input.reshape(x * y, z).float(), 0, 1), dim=0)
