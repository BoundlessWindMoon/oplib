import torch
from abc import ABC, abstractmethod


class Op(ABC):
    def __init__(self, name, backend, version,**kwargs):
        self.name = name
        self.backend = backend
        self.version = version
        self.kwargs = kwargs
        self.validate_backend(self.backend)
        
    @abstractmethod
    def get_func(self):
        pass    
        
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def prepare_data(self):
        pass

    def validate_backend(self, backend):
        backend_requirement = {
            "eager": {"module": None},
            "cuda": {
                "module": "torch.cuda",
                "error_msg": "CUDA backend not supported on your env",
            },
            "triton": {
                "module": "triton",
                "error_msg": "triton backend not supported on your env",
            },
            "cutile": {
                "module": "cutile",
                "error_msg": "cutile backend not supported on your env",
            },
            "tilelang": {
                "module": "tilelang",
                "error_msg": "tilelang backend not supported on your env",
            },
        }
        
        if backend not in backend_requirement:
            raise ValueError(f"Unsupported backend: {backend}. "
                             f"Supported backends: {list(backend_requirement.keys())}")
            
        req = backend_requirement[backend]
        if req["module"] is not None:
            try:
                __import__(req["module"])
            except ImportError as e:
                raise ImportError(req["error_msg"]) from e
            
        
