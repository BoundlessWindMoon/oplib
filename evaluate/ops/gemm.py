import torch
from evaluate.op import Op


class GemmOp(Op):
    def __init__(self, name, backend,device):
        self.M = 4096
        self.N = 4096
        self.K = 4096
        self.name = name
        self.backend = backend
        self.device = device
        self.dtype = torch.half
        self.A: torch.Tensor
        self.B: torch.Tensor
        self.C: torch.Tensor

    def prepare_data(self):
        M = self.M
        N = self.N
        K = self.K
        self.A = torch.randn(M, K, dtype=self.dtype, device=self.device)
        self.B = torch.randn(K, N, dtype=self.dtype, device=self.device)
        self.C = torch.randn(M, N, dtype=self.dtype, device=self.device)

    def get_reference(self):
        return self.run("eager")

    def get_result(self):
        return self.run(self.backend)

    def run(self, backend):
        if backend == "eager":
            return self.C + self.A @ self.B
        elif backend == "cuda":
            import gemm
            return gemm.gemm(self.A, self.B, self.C, self.M, self.N, self.K)
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
