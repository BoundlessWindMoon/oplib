# utils/reporter.py
from tqdm import tqdm
from collections import defaultdict

class ProgressReporter:
    @staticmethod
    def report(results_iter, total=None):
        results = []
        
        with tqdm(
            results_iter,
            total=total,
            desc="Running tests",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            dynamic_ncols=True
        ) as pbar:
            for result in pbar:
                pbar.set_postfix_str(
                    f"{result['ctx'].op_instance.name}"
                    f"({result['ctx'].op_instance.backend})"
                )
                results.append(result)
        
        print("\n=== FINAL RESULTS ===")
        print("=" * 80)
        
        by_op = defaultdict(list)
        for r in results:
            by_op[r['ctx'].op_instance.name].append(r)
        
        for op_name, op_results in by_op.items():
            print(f"\n[OPERATION] {op_name.upper()}")
            print("-" * 40)
            for r in op_results:
                print(ProgressReporter._format_result(r))
        
        print("\n" + "=" * 80)
        return results

    @staticmethod
    def _format_result(result):
        ctx = result['ctx']
        backend = f"{ctx.op_instance.backend}.{ctx.op_instance.version}"
        
        if 'error' in result:
            return f"✗ {backend.ljust(12)} | ERROR: {result['error']}"
        
        if 'ref_time' in result:
            ratio = result['ref_time'] / result['op_time']
            color = "\033[32m" if ratio >= 1 else "\033[33m"
            return (
                f"✓ {backend.ljust(12)} | "
                f"Ratio: {color}{ratio:>6.2f}x\033[0m | "
                f"Eager: {result['ref_time']:>7.3f}ms | "
                f"Custom: {result['op_time']:>7.3f}ms"
            )
        else:
            color = "\033[32m"
            return (
                f"? {backend.ljust(12)} | "
                f"Custom: {result['op_time']:>7.3f}ms"
            )
        return f"• {backend.ljust(12)} | Time: {result['op_time']:>7.3f}ms"
