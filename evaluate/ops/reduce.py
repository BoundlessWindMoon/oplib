import torch
from evaluate.op import Op


class ReduceOp(Op):
    def __init__(self, name, backend, version, device):
        super().__init__(name=name, backend=backend, version=version)
        self.N = 102410240
        self.name = name
        self.backend = backend
        self.device = device
        self.X: torch.Tensor
        self.SUM: torch.Tensor

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
                from reduce import reduce_v0

                self._backend_impls[backend][version] = reduce_v0

        elif backend == "triton":
            if version == "v0":
                from backend.triton.ops.reduce import reduce_v0

                self._backend_impls[backend][version] = reduce_v0
                
        elif backend == "tilelang":
            if version == "v0":
                from backend.tilelang.ops.reduce import reduce_v0

                self._backend_impls[backend][version] = reduce_v0

    def prepare_data(self):
        self.X = torch.randn(self.N, device=self.device)
        self.SUM = torch.zeros(1, device=self.device)

    def get_reference(self):
        return self.run("eager")

    def get_result(self):
        return self.run(self.backend, self.version)

    def run(self, backend, version=None):
        if backend == "eager":
            return torch.sum(self.X)
        else:
            try:
                func = self.get_func(backend, version)
                if func is None:
                    raise ImportError(f"No implementation found fo {backend}/{version}")
                self.SUM = func(self.X, self.SUM)
                return self.SUM[0]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to execute {backend}/{version}: {str(e)}"
                )

        # elif backend == "cuda":
        #     import reduce

        #     self.SUM = reduce.reduce(self.X, self.SUM)
        #     return self.SUM[0]

        # elif backend == "triton":
        #     raise NotImplementedError(
        #         f"{self.name}: triton backend not implemented yet"
        #     )
        # elif backend == "tilelang":
        #     raise NotImplementedError(
        #         f"{self.name}: tilelang backend not implemented yet"
        #     )
        # else:
        #     raise ValueError(f"{self.name}: backend not implemented ye")
