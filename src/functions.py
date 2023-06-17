import torch
import torchvision.transforms as T

from scipy.stats import norm
from torch.distributions.normal import Normal

scipy_cdf = T.Lambda(lambda x: torch.Tensor(norm.cdf(x.detach().cpu())).to(x.device))
scipy_ppf = T.Lambda(lambda x: torch.Tensor(norm.ppf(x.detach().cpu())).to(x.device))

def cdf(
        x: torch.Tensor
    ) -> torch.Tensor:
    return Normal(0., 1.).cdf(x)

def polynomial_function_evaluator(
        x: torch.Tensor,
        coef: torch.Tensor
    ) -> torch.Tensor:
    """
    Evaluates the polynomial described by the coefficients stored in <coef> 
    given the tensor <x>.

    Source:
    Holin et. al., "Polynomial and Rational Function Evaluation"
    https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/rational.html
    """
    device = x.device
    accum  = torch.tensor(0).to(device)
    for c in coef:
        accum = x * accum + c
    return accum

_polynomial_function_evaluator = T.Lambda(lambda iota: polynomial_function_evaluator(*iota))