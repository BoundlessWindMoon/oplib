import torch
from evaluate.op import Op


class GemmOp(Op):
    def __init__(self, name, backend, version, device):
        super().__init__(name=name, backend=backend, version=version)
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

        self._backend_impls = {
            "cuda": {
                "v0": None,
            },
            "triton": {
                "v0": None,
            },
            "tilelang": {
                "v0": None,
            },
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
                from gemm import gemm_v0

                self._backend_impls[backend][version] = gemm_v0

        elif backend == "triton":
            if version == "v0":
                from backend.triton.ops.gemm import gemm_v0

                self._backend_impls[backend][version] = gemm_v0
                
        elif backend == "tilelang":
            if version == "v0":
                from backend.tilelang.ops.gemm import gemm_v0

                self._backend_impls[backend][version] = gemm_v0

    def prepare_data(self):
        M = self.M
        N = self.N
        K = self.K
        self.A = torch.rand(M, K, dtype=self.dtype, device=self.device) - 0.5
        self.B = torch.rand(K, N, dtype=self.dtype, device=self.device) - 0.5
        self.C = torch.rand(M, N, dtype=self.dtype, device=self.device) -0.5

    def get_reference(self):
        return self.run("eager")

    def get_result(self):
        return self.run(self.backend, self.version)

    def run(self, backend, version=None):
        if backend == "eager":
            return self.C + self.A @ self.B
        else:
            try:
                func = self.get_func(backend, version)
                if func is None:
                    raise ImportError(f"No implementation found fo {backend}/{version}")
                output = func(self.A, self.B, self.C, self.M, self.N, self.K)
                return output
            except Exception as e:
                raise RuntimeError(
                    f"Failed to execute {backend}/{version} vadd: {str(e)}"
                )
