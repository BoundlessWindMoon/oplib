import torch
from evaluate.op import Op


class VaddOp(Op):
    def __init__(self, name, backend, device):
        super().__init__(name, backend)
        self.name = name
        self.backend = backend
        self.device = device

        # Ignore incx and incy
        self.N = 102400
        self.X: torch.Tensor
        self.Y: torch.Tensor
        self.incx = 1
        self.incy = 1

    def prepare_data(self):
        self.X = torch.randn(self.N, device=self.device)
        self.Y = torch.randn(self.N, device=self.device)

    def get_reference(self):
        reference = self.run("eager")
        return reference

    def get_result(self):
        result = self.run(self.backend)
        return result

    def eval(self):
        reference_result = self.run("eager")
        result = self.run(self.backend)
        error = torch.abs(reference_result - result).max().item()
        print(f"max_error: {error:.9f}")
        return error < 1e-5

    def run(self, backend):
        if backend == "eager":
            assert len(self.X) == len(
                self.Y
            ), f"Length mismatch: X({len(self.X)}) != Y({len(self.Y)})"
            assert self.N <= len(self.X), f"{self.N} exceeds X length({len(self.X)})"
            assert self.N <= len(self.Y), f"{self.N} exceeds Y length({len(self.Y)})"

            if self.incx == 1 and self.incy == 1:
                Z = self.X[: self.N] + self.Y[: self.N]
            else:
                raise NotImplementedError("(incx > 1 or incy > 1) not implemented yet")
            return Z

        elif backend == "triton":
            raise NotImplementedError("Triton backend not implemented yet")
        elif backend == "cuda":
            import vector_add

            Z = vector_add.add(self.X, self.Y)
            return Z
        else:
            raise ValueError(f"Unknown backend: {backend}")
