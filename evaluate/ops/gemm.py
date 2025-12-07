import torch
from evaluate.op import Op


class GemmOp(Op):
    def __init__(self, name, backend, device):
        self.M = 4096
        self.N = 4096
        self.K = 4096
        self.name = name
        self.backend = backend
        self.device = device
        self.A: torch.Tensor
        self.B: torch.Tensor
        self.C: torch.Tensor

    def prepare_data(self):
        M = self.M
        N = self.N
        K = self.K
        self.A = torch.randn(M, K)
        self.B = torch.randn(K, N)
        self.C = torch.randn(M, N)

    def get_reference(self):
        return self.run("eager")

    def get_result(self):
        return self.run(self.backend)

    def run(self, backend):
        if backend == "eager":
            return self.C + self.A @ self.B
        elif backend == "cuda":
            raise NotImplementedError(f"{self.name}: cuda backend not implemented yet")
        elif backend == "triton":
            raise NotImplementedError(
                f"{self.name}: triton backend not implemented yet"
            )
        elif backend == "tilelang":
            raise NotImplementedError(
                f"{self.name}: tilelang backend not implemented yet"
            )
        else:
            raise ValueError(f"{self.name}: backend not implemented yet")
