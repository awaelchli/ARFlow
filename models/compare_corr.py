from models.correlation_native import Correlation
from spatial_correlation_sampler import spatial_correlation_sample

_search_range = 4
_corr = Correlation(max_displacement=_search_range)


def corr_native(x0, x1):
    return _corr(x0, x1)


def corr_spatial(x0, x1):
    out_corr = spatial_correlation_sample(
        input1=x0.contiguous(), input2=x1.contiguous(),
        kernel_size=1, patch_size=(_search_range * 2 + 1), stride=1, padding=0, dilation_patch=1
    )
    out_corr /= x0.shape[1]  # native takes mean, so need to divide by channels
    out_corr = out_corr.flatten(1, 2)
    return out_corr


if __name__ == "__main__":
    import torch
    # torch.manual_seed(100)
    x0 = torch.ones(2, 3, 64, 64, requires_grad=True).to(torch.device('cuda', 1))
    x1 = torch.ones(2, 3, 64, 64, requires_grad=True).to(torch.device('cuda', 1))

    x0_c = torch.as_tensor(x0)
    x0_c.retain_grad()
    x1_c = torch.as_tensor(x1)
    x1_c.retain_grad()

    c0 = corr_native(x0, x1)
    c1 = corr_spatial(x0_c, x1_c)

    print(c0.mean())
    print(c1.mean())

    c0.sum().backward(retain_graph=True)
    c1.sum().backward(retain_graph=True)

    print(x0_c.requires_grad)
    print((x0.grad - x0_c.grad).abs().sum())
    print((x1.grad - x1_c.grad).abs().sum())

    err = (c0 -
            c1).abs().sum()
    print(err)
