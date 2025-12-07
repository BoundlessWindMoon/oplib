import torch
from evaluate.ops.attention import AttentionOp
from evaluate.ops.gemm import GemmOp
from evaluate.ops.reduce import ReduceOp
from evaluate.ops.vector_add import VaddOp
from configparser import ConfigParser


class Evaluator:
    class OpContext:
        def __init__(self, op_class, num_warmup=5, num_eval=5, device="cpu"):
            self.op_class = op_class
            self.op_instance = None
            self.num_warmup = num_warmup
            self.num_eval = num_eval
            self.device = device

    def __init__(
        self, config_path="./config/ops.ini", device="cpu"
    ):
        self.parser = ConfigParser()
        readed_configs = self.parser.read(config_path)
        if not readed_configs:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config_path = config_path
        self.device = device

        self.op_ctxs = []
        self.op_registry = {}

    def register(self):
        self.register_op_type()
        self.register_ops()

    def register_op_type(self):
        # TODO: delete hard code
        self.op_registry = {
            "reduce": ReduceOp,
            "gemm": GemmOp,
            "attention": AttentionOp,
            "vadd": VaddOp,
        }

    def register_ops(self):
        parser = self.parser
        sections = parser.sections()
        for section in sections:
            name = parser.get(section, "name")
            backend = parser.get(section, "backend")
            num_warmup = self.parser.getint(section, "num_warmup", fallback=5)
            num_eval = self.parser.getint(section, "num_eval", fallback=5)

            if op_class := self.op_registry.get(name):
                # create ctx by op_class
                ctx = self.OpContext(
                    op_class=op_class,
                    num_warmup=num_warmup,
                    num_eval=num_eval,
                    device=self.device,
                )
                ctx.op_instance = op_class(name, backend, device=ctx.device)
                self.op_ctxs.append(ctx)
            else:
                raise ValueError(f"Unregistered operation:{name}")

    def get_op_ctxs(self):
        return self.op_ctxs

    def eval(self, ctx):
        op = ctx.op_instance
        num_warmup = ctx.num_warmup
        num_eval = ctx.num_eval

        # prepare for test data
        op.prepare_data()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        reference = op.get_reference()
        result = op.get_result()
        error = torch.abs(reference - result).max().item()
        assert error < 1e-5, f"op {op.name} error! max_error = {error}"
        assert torch.cuda.is_available(), "torch.cuda.is_available == False"

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
