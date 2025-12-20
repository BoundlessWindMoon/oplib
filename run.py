from evaluate.evaluator import Evaluator
from evaluate.op import Op
import argparse

config_path = "./config/ops.ini"
device = "cuda:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default=config_path)
    args = parser.parse_args()

    evaluator = Evaluator(config_path=args.c, device=device)
    evaluator.parse_config()
    ctxs = evaluator.get_op_ctxs()
    run_eval = evaluator.get_eval_info()

    for ctx in ctxs:
        if run_eval is True:
            reference_time, op_time = evaluator.run_full_evaluation(ctx)
            print(
                f"Result of op({ctx.op_instance.name}) PASSED! accelerate ratio = {reference_time / op_time}"
            )
            print(f"reference(backend=eager): time = {reference_time}")
            print(f"custom op(backend={ctx.op_instance.backend}, version={ctx.op_instance.version}): time = {op_time}\n")
        else:
            op_time = evaluator.run_custom_ops(ctx)
            print(f"custom op(backend={ctx.op_instance.backend}, version={ctx.op_instance.version}): time = {op_time}\n")


if __name__ == "__main__":
    main()
