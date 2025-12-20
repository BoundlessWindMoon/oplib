from evaluate.evaluator import Evaluator
from evaluate.op import Op
from utils.reporter import ProgressReporter
import argparse

config_path = "./config/ops.toml"
device = "cuda:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default=config_path)
    args = parser.parse_args()

    evaluator = Evaluator(config_path=args.c, device=device)
    evaluator.parse_config()
    ctxs = evaluator.get_op_ctxs()

    raw_results = evaluator.evaluate_ops(
        evaluator.get_op_ctxs(),
        evaluator.get_eval_info()
    )
    
    print("\n=== Evaluation Results ===")
    for formatted_result in ProgressReporter.report(raw_results, total=len(ctxs)):
        print(formatted_result)


if __name__ == "__main__":
    main()
