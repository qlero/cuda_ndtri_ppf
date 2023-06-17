
import torch

from .constants import SQUARE_ROOT_PI
from .constants import CHECK_1_MINUS_EXPMIN2, CHECK_EXPMIN2
from .constants import APPROX_INTERVAL_Y_MIN_HALF_1, APPROX_INTERVAL_Y_MIN_HALF_2
from .constants import APPROX_INTERVAL_SQRT_2LOGY2to8_1, APPROX_INTERVAL_SQRT_2LOGY2to8_2
from .constants import APPROX_INTERVAL_SQRT_2LOGY8to64_1, APPROX_INTERVAL_SQRT_2LOGY8to64_2

from .functions import _polynomial_function_evaluator

def ppf(
        x: torch.Tensor
    ) -> torch.Tensor:
    """
    Function equivalent to scipy.stats.norm.ppf or scipy.special.cephes.ndtri.

    Inverse of the Standard normal CDF function.
    """

    # returns an error is x in ]-inf, 0] or [1, +inf[
    # NOTE: difference with ndtri.c implementation that returns -inf or +inf
    # if x == 0 and x == 1 respectively
    if torch.any(torch.logical_and(x<=0, x>=1)):
        raise ValueError("All scalars must be in ]0, 1[ range.")
    
    # retrieves device to use
    device = x.device

    # Instantiates constants & variables
    shape  = x.shape
    S2PI   = SQUARE_ROOT_PI.to(device)
    AIY1   = APPROX_INTERVAL_Y_MIN_HALF_1.to(device)
    AIY2   = APPROX_INTERVAL_Y_MIN_HALF_2.to(device)
    AIS11  = APPROX_INTERVAL_SQRT_2LOGY2to8_1.to(device)
    AIS12  = APPROX_INTERVAL_SQRT_2LOGY2to8_2.to(device)
    AIS21  = APPROX_INTERVAL_SQRT_2LOGY8to64_1.to(device)
    AIS22  = APPROX_INTERVAL_SQRT_2LOGY8to64_2.to(device)
    NEGATE = (torch.ones(shape) == 1).to(device)

    # Step 1.
    CHECK         = x > CHECK_1_MINUS_EXPMIN2
    x[CHECK]      = 1. - x[CHECK]
    NEGATE[CHECK] = False

    # Step 2.
    CHECK2     = x > CHECK_EXPMIN2
    x[CHECK2]  = x[CHECK2] - 0.5
    temp       = x[CHECK2] * x[CHECK2]
    pol_temp_1 = _polynomial_function_evaluator([temp, AIY1])
    pol_temp_2 = _polynomial_function_evaluator([temp, AIY2])
    x[CHECK2]  = x[CHECK2] + x[CHECK2] * (temp * pol_temp_1 / pol_temp_2)
    x[CHECK2]  = S2PI * x[CHECK2]

    # Step 3.
    CHECK  = ~CHECK2
    NEGATE = torch.logical_and(CHECK, NEGATE)
    temp   = torch.sqrt(-2. * torch.log(x[CHECK]))
    temp2  = temp - torch.log(temp) / temp
    temp3  = torch.tensor(1.) / temp

    # Step 4.
    CHECK2         = temp < torch.tensor(0.8)
    temp4          = torch.zeros(temp.shape).to(temp.dtype).to(device)
    pol_temp_1     = _polynomial_function_evaluator([temp3[CHECK2], AIS11])
    pol_temp_2     = _polynomial_function_evaluator([temp3[CHECK2], AIS12])
    temp4[CHECK2]  = temp3[CHECK2] * pol_temp_1 / pol_temp_2
    pol_temp_1     = _polynomial_function_evaluator([temp3[~CHECK2], AIS21])
    pol_temp_2     = _polynomial_function_evaluator([temp3[~CHECK2], AIS22])
    temp4[~CHECK2] = temp3[~CHECK2] * pol_temp_1 / pol_temp_2
    temp           = temp2 - temp4

    # Step 5.
    x[CHECK]  = temp
    x[NEGATE] = -1. * x[NEGATE]

    return x