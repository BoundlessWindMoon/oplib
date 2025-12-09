from evaluate.evaluator import Evaluator
from evaluate.op import Op

config_path = "./config/ops.ini"
device = "cuda:0"


def main():
    evaluator = Evaluator(config_path=config_path, device=device)
    evaluator.register()
    ctxs = evaluator.get_op_ctxs()
    for ctx in ctxs:
        reference_time, op_time = evaluator.eval(ctx)
        print(
            f"Result of op({ctx.op_instance.name}) PASSED! accelerate ratio = {reference_time / op_time}"
        )
        print(f"reference(backend=eager): time = {reference_time}")
        print(f"custom op(backend={ctx.op_instance.backend}): time = {op_time}\n")


if __name__ == "__main__":
    main()
