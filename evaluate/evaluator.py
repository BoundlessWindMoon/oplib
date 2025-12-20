import torch
import tomllib
from evaluate.ops.attention import AttentionOp
from evaluate.ops.gemm import GemmOp
from evaluate.ops.reduce import ReduceOp
from evaluate.ops.vector_add import VaddOp
from configparser import ConfigParser
from tqdm import tqdm

class Evaluator:
    class OpContext:
        def __init__(
            self,
            op_class,
            op_version,
            num_warmup=5,
            num_eval=5,
            tolerance=1e-5,
            device="cpu",
        ):
            self.op_class = op_class
            self.op_instance = None
            self.op_version = op_version
            self.num_warmup = num_warmup
            self.num_eval = num_eval
            self.tolerance = tolerance
            self.device = device

    def __init__(self, config_path="./config/ops.toml", device="cpu"):
        self.device = device
        self.run_eval = False
        self.op_ctxs = []
        self.op_registry = {}
        
        try:
            with open(config_path, "rb") as f:
                self.config = tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except Exception as e:
            raise ValueError(f"Failed to parse TOML config: {str(e)}")

    def parse_config(self):
        self.register_op_type()
        self.parse()

    def register_op_type(self):
        # TODO: delete hard code
        self.op_registry = {
            "reduce": ReduceOp,
            "gemm": GemmOp,
            "attention": AttentionOp,
            "vadd": VaddOp,
        }

    def get_eval_info(self):
        return self.run_eval

    def parse(self):
        self.run_eval = self.config.get("global", {}).get("run_eval", False)
        
        ops_config = self.config.get("ops", {})
        
        with tqdm(
            total=sum(len(op_conf.get("impl", [])) for op_conf in ops_config.values()),
            desc="Parsing ops",
            unit="impl"
        ) as pbar:
            
            for short_name, op_conf in ops_config.items():
                num_warmup = op_conf.get("num_warmup", 5)
                num_eval = op_conf.get("num_eval", 5)
                tolerance = op_conf.get("tolerance", 1e-5)
                
                if (op_class := self.op_registry.get(short_name)) is None:
                    raise ValueError(f"Unregistered operation: {short_name}")
                
                for impl in op_conf.get("impl", []):
                    backend = impl["backend"]
                    for version in impl["versions"]:
                        ctx = self.OpContext(
                            op_class=op_class,
                            op_version=version,
                            num_warmup=num_warmup,
                            num_eval=num_eval,
                            tolerance=tolerance,
                            device=self.device,
                        )
                        ctx.op_instance = op_class(short_name, backend, version, device=ctx.device)
                        self.op_ctxs.append(ctx)
                    
                    pbar.update(1)
                    pbar.set_postfix(op=short_name, backend=backend)
                    
    def get_op_ctxs(self):
        return self.op_ctxs

    def evaluate_ops(self, ctxs, run_eval):
        for ctx in ctxs:
            try:
                if run_eval:
                    ref_time, op_time = self.run_full_evaluation(ctx)
                    yield {'ctx': ctx, 'ref_time': ref_time, 'op_time': op_time}
                else:
                    op_time = self.run_custom_ops(ctx)
                    yield {'ctx': ctx, 'op_time': op_time}
            except Exception as e:
                yield {'ctx': ctx, 'error': str(e)}
    
    def run_custom_ops(self, ctx):
        op = ctx.op_instance
        num_warmup = ctx.num_warmup
        num_eval = ctx.num_eval

        # prepare for test data
        op.prepare_data()

        # warmup
        for _ in range(num_warmup):
            op.get_result()

        def measure_time(func):
            """Helper function to measure execution time"""
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()

            for _ in range(num_eval):
                func()

            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / num_eval

        # Measure performance
        op_time = measure_time(op.get_result)

        return {op_time / num_eval}

    def run_full_evaluation(self, ctx):
        assert torch.cuda.is_available(), "torch.cuda.is_available == False"
        
        op = ctx.op_instance
        num_warmup = ctx.num_warmup
        num_eval = ctx.num_eval

        # prepare for test data
        op.prepare_data()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        reference = op.get_reference()
        result = op.get_result()
        # error = torch.abs((reference - result)).mean().item()
        torch.testing.assert_close(reference, result, rtol= ctx.tolerance, atol= ctx.tolerance)

        # warmup
        for _ in range(num_warmup):
            op.get_result()
            op.get_reference()

        def measure_time(func):
            """Helper function to measure execution time"""
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()

            for _ in range(num_eval):
                func()

            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / num_eval

        # Measure performance
        reference_time = measure_time(op.get_reference)
        op_time = measure_time(op.get_result)

        return {reference_time / num_eval, op_time / num_eval}
