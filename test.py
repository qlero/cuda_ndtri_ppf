import numpy as np
import time
import torch
import torchvision.transforms as T

from src import scipy_ppf
from src import scipy_cdf
from src import torch_cdf
from src import torch_ppf

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__ == "__main__":

    with torch.no_grad():

        start = time.time()
        for _ in range(50):
            x = torch.randn(1000,1000).detach().to(device)
            _ = np.all(np.round((scipy_ppf(scipy_cdf(x)) - x).detach().cpu().numpy(), 1)==0)
        end = time.time()

        print(f"scipy-based process ran in {np.round(end-start, 4)}s")

        start = time.time()
        for _ in range(50):
            x = torch.randn(1000,1000).detach().to(device)
            _ = np.all(np.round((torch_ppf(torch_cdf(x)) - x).detach().cpu().numpy(), 1)==0)
        end = time.time()

        print(f"Pytorch-based process ran in {np.round(end-start, 4)}s")
