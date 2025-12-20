import torch
import torch.nn as nn
import math
from evaluate.op import Op
from torch.nn import functional as F

class AttentionOp(Op):
    def __init__(self, name, backend, version, device):
        super().__init__(name=name, backend=backend, version=version)
        self.name = name
        self.backend = backend
        self.device = device
        self.X = torch.Tensor
        self.W_q = torch.Tensor
        self.W_k = torch.Tensor
        self.W_v = torch.Tensor

        # llama-7B param
        self.B = 16  # Batchsize
        self.S = 64  # Seq_len
        self.H = 12  # Num_head
        self.HD = 64  # Head_dim
        self.D = 768  # Hidden_dim

        self.dtype = torch.float
        
        self._backend_impls = {
            "cuda": {
                "v0": None,
            },
            "triton": {
                "v0": None,
            }
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
                from attention import attn_v0
                self._backend_impls[backend][version] = attn_v0
            
        elif backend == "triton":
            if version == "v0":
                from backend.triton.ops.attention import attn_v0 
                self._backend_impls[backend][version] = attn_v0

    def prepare_data(self):
        B = self.B
        S = self.S
        H = self.H
        HD = self.HD
        D = self.D
        
        self.Q = torch.randn(B, H, S, HD).cuda()
        self.K = torch.randn(B, H, S, HD).cuda()
        self.V = torch.randn(B, H, S, HD).cuda()
        
        
    def get_reference(self):
        return self.run("eager")

    def get_result(self):
        return self.run(self.backend, self.version)

    def run(self, backend="eager", version=None):
        B = self.B
        S = self.S
        H = self.H
        HD = self.HD
        D = self.D

        X = self.X
        W_q = self.W_q
        W_k = self.W_k
        W_v = self.W_v
        Q = self.Q
        K = self.K
        V = self.V
        
        if backend == "eager":
            with torch.no_grad():

                # [B, H, S, HD] @ [B, H, HD, S] -> [B, H, S, S]
                attn_score = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(K.size(-1)))
                attn_score = F.softmax(attn_score, dim=-1)

                # [B, H, S, S] @ [B, H, S, HD] -> [B, H, S, HD]
                output = attn_score @ V

                # [B, H, S, HD] -> [B, S, H, HD] -> [B, S, D]
                output = output.transpose(1, 2).contiguous().view(B, S, D)
                return output
        else:
            try:
                func = self.get_func(backend, version)
                if func is None:
                    raise ImportError(f"No implementation found fo {backend}/{version}")
                output = func(Q, K, V)
                return output.transpose(1, 2).contiguous().view(B, S, D)
            except Exception as e:
                raise RuntimeError(f"Failed to execute {backend}/{version} attention: {str(e)}")
            
