# Benchmarks

## CUDA-Q MPS

`benchmark_cudaq_mps.py` compares the existing CPU `MPSEvaluator` with the optional
CUDA-Q `CudaQMPSBenchmarkEvaluator` using NVIDIA CUDA-Q's `tensornet-mps` target.
This is a first benchmark backend for QAOA MaxCut, not the final optimized custom
cuTensorNet evaluator.

CUDA-Q is optional. To run this benchmark, install CUDA-Q on a supported platform
with an NVIDIA GPU, then run for example:

```bash
python benchmarks/benchmark_cudaq_mps.py --n 100 --p 1 --graph-type random-regular --max-bond 64
```

The CUDA-Q evaluator returns the same energy convention as the rest of the package:
for graph inputs, MaxCut is represented as `sum -0.5 * weight * Z_i Z_j`, and the
reported value is `<H_C>`.

## cuQuantum MPS

`CuQuantumMPSEvaluator` is an experimental optional backend that builds the default
QAOA state directly with cuTensorNet's MPS `NetworkState` API and evaluates the
energy with one cached `NetworkOperator` built directly from the `SparsePauliOp`
terms using cuQuantum product-operator components. It avoids both the high-level
CUDA-Q `observe` interface used by the CUDA-Q benchmark backend and the CPU Quimb
MPO construction path used by `MPSEvaluator`.

This evaluator is intentionally narrower than `MPSEvaluator`: it currently supports
real diagonal one- and two-local Z cost operators, the default |+> initial state,
and the standard X mixer. It requires a supported cuQuantum Python installation,
for example `pip install cuquantum-python-cu12` on supported CUDA 12 platforms.
