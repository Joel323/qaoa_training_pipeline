#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Benchmark CUDA-Q ``tensornet-mps`` against the CPU MPS evaluator.

Example:
    python benchmarks/benchmark_cudaq_mps.py --n 100 --p 1 --graph-type random-regular

CUDA-Q is optional and requires a supported CUDA-Q installation plus an NVIDIA GPU for
the ``tensornet-mps`` target.
"""

import argparse
import math
import time
from collections.abc import Callable

import networkx as nx
import numpy as np

from qaoa_training_pipeline.evaluation import MPSEvaluator
from qaoa_training_pipeline.evaluation.cudaq_mps import CudaQMPSBenchmarkEvaluator
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Number of graph nodes.")
    parser.add_argument("--p", type=int, default=1, help="QAOA depth.")
    parser.add_argument(
        "--graph-type",
        choices=["path", "cycle", "random-regular"],
        default="path",
        help="Graph family to benchmark.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Degree for --graph-type random-regular.",
    )
    parser.add_argument(
        "--max-bond",
        type=int,
        default=64,
        help="Maximum MPS bond dimension for CUDA-Q and CPU MPS.",
    )
    parser.add_argument(
        "--abs-cutoff",
        type=float,
        default=1.0e-5,
        help="Absolute MPS truncation cutoff.",
    )
    parser.add_argument(
        "--relative-cutoff",
        type=float,
        default=1.0e-5,
        help="CUDA-Q relative MPS truncation cutoff.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp64", "fp32"],
        default="fp64",
        help="CUDA-Q MPS precision option.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Warm repeated evaluations per evaluator.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def _build_graph(args: argparse.Namespace) -> nx.Graph:
    """Build the requested benchmark graph."""

    if args.n <= 1:
        raise ValueError("--n must be greater than 1.")

    if args.graph_type == "path":
        return nx.path_graph(args.n)

    if args.graph_type == "cycle":
        return nx.cycle_graph(args.n)

    if args.degree <= 0 or args.degree >= args.n:
        raise ValueError("--degree must satisfy 0 < degree < n for random-regular graphs.")
    if (args.degree * args.n) % 2 != 0:
        raise ValueError("degree * n must be even for random-regular graphs.")

    return nx.random_regular_graph(d=args.degree, n=args.n, seed=args.seed)


def _time_evaluator(
    evaluate: Callable[[], float], repeats: int
) -> tuple[float, float, list[float]]:
    """Return the energy, cold runtime, and warm runtimes for an evaluator."""

    start = time.perf_counter()
    energy = evaluate()
    cold_runtime = time.perf_counter() - start

    warm_runtimes = []
    for _ in range(repeats):
        start = time.perf_counter()
        energy = evaluate()
        warm_runtimes.append(time.perf_counter() - start)

    return energy, cold_runtime, warm_runtimes


def _format_seconds(value: float) -> str:
    """Format runtime values compactly."""

    return f"{value:.6f}s"


def main() -> None:
    """Run the benchmark."""

    args = _parse_args()
    graph = _build_graph(args)
    cost_op = graph_to_operator(graph, pre_factor=-0.5)

    rng = np.random.default_rng(args.seed)
    params = rng.uniform(-np.pi / 4, np.pi / 4, size=2 * args.p).tolist()

    cpu_evaluator = MPSEvaluator(
        threshold_circuit=args.abs_cutoff,
        bond_dim_circuit=args.max_bond,
    )
    cudaq_evaluator = CudaQMPSBenchmarkEvaluator(
        max_bond_dim=args.max_bond,
        abs_cutoff=args.abs_cutoff,
        relative_cutoff=args.relative_cutoff,
        precision=args.precision,
    )

    repeats = max(0, args.repeats)
    cpu_energy, cpu_cold, cpu_warm = _time_evaluator(
        lambda: cpu_evaluator.evaluate(cost_op, params),
        repeats,
    )
    cudaq_energy, cudaq_cold, cudaq_warm = _time_evaluator(
        lambda: cudaq_evaluator.evaluate(cost_op, params),
        repeats,
    )

    abs_error = abs(cudaq_energy - cpu_energy)
    rel_error = math.nan if abs(cpu_energy) <= 1.0e-15 else abs_error / abs(cpu_energy)

    print("CUDA-Q MPS benchmark")
    print(f"graph_type: {args.graph_type}")
    print(f"n_nodes: {graph.number_of_nodes()}")
    print(f"n_edges: {graph.number_of_edges()}")
    print(f"qaoa_depth_p: {args.p}")
    print(f"max_bond_dimension: {args.max_bond}")
    print(f"abs_cutoff: {args.abs_cutoff}")
    print(f"relative_cutoff: {args.relative_cutoff}")
    print(f"precision: {args.precision}")
    print(f"cpu_energy: {cpu_energy:.12g}")
    print(f"cudaq_energy: {cudaq_energy:.12g}")
    print(f"absolute_error: {abs_error:.12g}")
    print(f"relative_error: {rel_error:.12g}")
    print(f"cpu_cold_runtime: {_format_seconds(cpu_cold)}")
    print(f"cudaq_cold_runtime: {_format_seconds(cudaq_cold)}")

    if cpu_warm:
        print(f"cpu_warm_mean_runtime: {_format_seconds(float(np.mean(cpu_warm)))}")
        print(f"cpu_warm_min_runtime: {_format_seconds(min(cpu_warm))}")
    if cudaq_warm:
        print(f"cudaq_warm_mean_runtime: {_format_seconds(float(np.mean(cudaq_warm)))}")
        print(f"cudaq_warm_min_runtime: {_format_seconds(min(cudaq_warm))}")


if __name__ == "__main__":
    main()
