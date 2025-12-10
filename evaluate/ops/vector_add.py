import torch
from evaluate.op import Op


class VaddOp(Op):
    def __init__(self, name, backend, version, device):
        super().__init__(name=name, backend=backend, version=version)
        self.name = name
        self.backend = backend
        self.device = device

        # Ignore incx and incy
        self.N = 102400
        self.X: torch.Tensor
        self.Y: torch.Tensor
        self.incx = 1
        self.incy = 1

        self._backend_impls = {
            "cuda": {
                "v0": None,
                "v1": None,
            },
            "triton": {"v0": None},
        }

    def get_func(self, backend, version):
        if backend not in self._backend_impls:
            raise ValueError(f"Unsupported backend {backend}")
        if version not in self._backend_impls[backend]:
            raise ValueError(f"Unsupported verison {version} on backend {backend}")
        if self._backend_impls[backend][version] is None:
            self._load_implementation(backend, version)

        return self._backend_impls[backend][version]

    def _load_implementation(self, backend, version):
        if backend == "cuda":
            if version == "v0":
                from vector_add import vadd_v0

                self._backend_impls[backend][version] = vadd_v0

            elif version == "v1":
                from vector_add import vadd_v1

                self._backend_impls[backend][version] = vadd_v1

        elif backend == "triton":
            if version == "v0":
                from backend.triton.ops.vector_add import vadd_v0

                self._backend_impls[backend][version] = vadd_v0

    def prepare_data(self):
        self.X = torch.randn(self.N, device=self.device)
        self.Y = torch.randn(self.N, device=self.device)

    def get_reference(self):
        reference = self.run("eager")
        return reference

    def get_result(self, version=None):
        result = self.run(self.backend, self.version)
        return result

    def eval(self):
        reference_result = self.run("eager")
        result = self.run(self.backend)
        error = torch.abs(reference_result - result).max().item()
        print(f"max_error: {error:.9f}")
        return error < 1e-5

    def run(self, backend, version=None):
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

        else:
            try:
                func = self.get_func(backend, version)
                if func is None:
                    raise ImportError(f"No implementation found fo {backend}/{version}")
                output = func(self.X, self.Y)
                return output
            except Exception as e:
                raise RuntimeError(
                    f"Failed to execute {backend}/{version} vadd: {str(e)}"
                )
