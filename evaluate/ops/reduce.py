import torch
from evaluate.op import Op


class ReduceOp(Op):
    def __init__(self, name, backend, device):
        super().__init__(name, backend)
        self.N = 1048576
        self.name = name
        self.backend = backend
        self.device = device
        self.X: torch.Tensor
        self.SUM: torch.Tensor

    def prepare_data(self):
        self.X = torch.randn(self.N, device=self.device)
        self.SUM = torch.zeros(1, device=self.device)

    def get_reference(self):
        return self.run("eager")

    def get_result(self):
        return self.run(self.backend)

    def run(self, backend):
        if backend == "eager":
            return torch.sum(self.X)
        elif backend == "cuda":
            import reduce

            self.SUM = reduce.reduce(self.X, self.SUM)
            return self.SUM[0]

        elif backend == "triton":
            raise NotImplementedError(
                f"{self.name}: triton backend not implemented yet"
            )
        elif backend == "tilelang":
            raise NotImplementedError(
                f"{self.name}: tilelang backend not implemented yet"
            )
        else:
            raise ValueError(f"{self.name}: backend not implemented ye")
