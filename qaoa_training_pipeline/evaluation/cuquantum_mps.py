#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""cuQuantum MPS evaluator for diagonal QAOA cost Hamiltonians.

This module provides an optional NVIDIA cuQuantum/cuTensorNet backend that is
aimed at avoiding the per-Pauli-term overhead observed with high-level
``cudaq.observe`` on ``tensornet-mps``. The evaluator directly builds the QAOA
state with cuTensorNet ``NetworkState`` and computes the energy with one cached
``NetworkOperator`` built directly from the ``SparsePauliOp`` terms. This keeps
the evaluator on the cuQuantum path for both state evolution and energy
evaluation.

The first implementation intentionally targets the default QAOA/MaxCut-style path:
real diagonal one- and two-local Z terms, the |+> initial state, and the standard
X mixer. Custom mixers, custom initial states, custom ansatz circuits, and higher
order terms are rejected.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator


CUQUANTUM_OBSERVABLE_STRATEGIES = {"pauli_products"}


@dataclass(frozen=True)
class _DiagonalZTerm:
    """A real diagonal Z or ZZ cost term in qubit-index order."""

    qubits: tuple[int, ...]
    coefficient: float


def _require_cuquantum():
    """Import cuQuantum and raise a clear optional-dependency error if unavailable."""

    try:
        from cuquantum.tensornet.experimental import (  # pylint: disable=import-outside-toplevel
            MPSConfig,
            NetworkOperator,
            NetworkState,
        )
    except (ImportError, OSError) as exc:
        raise ImportError(
            "CuQuantumMPSEvaluator requires the optional cuQuantum Python package. "
            "Install the CUDA-version-specific package for your system, for example "
            "`pip install cuquantum-python-cu12` on supported CUDA 12 platforms."
        ) from exc

    return MPSConfig, NetworkOperator, NetworkState


class CuQuantumMPSEvaluator(BaseEvaluator):
    """Evaluate default QAOA energies with cuQuantum's MPS state simulator.

    Args:
        max_bond_dim: Maximum MPS bond dimension. This maps to cuTensorNet
            ``MPSConfig.max_extent``. ``None`` disables extent truncation.
        abs_cutoff: Absolute SVD truncation cutoff. ``None`` disables this cutoff.
        rel_cutoff: Relative SVD truncation cutoff. ``None`` disables this cutoff.
        precision: ``"fp64"`` maps to ``complex128`` and ``"fp32"`` maps to
            ``complex64``.
        svd_algo: cuTensorNet MPS SVD algorithm. Valid values are ``"gesvd"``,
            ``"gesvdj"``, ``"gesvdp"``, and ``"gesvdr"``. ``None`` leaves the
            package default unchanged.
        mpo_application: cuTensorNet MPS-MPO operation mode, either ``"exact"``,
            ``"approximate"``, or ``None`` to leave the package default.
        observable_strategy: ``"pauli_products"`` builds one cached cuQuantum
            ``NetworkOperator`` from identity-free Pauli product components. The
            option is kept explicit to leave room for a future cuQuantum-native
            compressed MPO strategy.
        device_id: Optional CUDA device id passed through ``NetworkOptions``.
        network_options: Optional dictionary passed as cuTensorNet network options.
            ``device_id`` overrides the same key if both are supplied.
        release_workspace: Whether to release cuTensorNet workspace after the
            expectation call.

    Limitations:
        This first version supports only real diagonal one- and two-local Z terms
        in the cost Hamiltonian and the default QAOA construction.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        max_bond_dim: int | None = 64,
        abs_cutoff: float | None = 1.0e-6,
        rel_cutoff: float | None = 1.0e-6,
        precision: str = "fp32",
        svd_algo: str | None = "gesvdj",
        mpo_application: str | None = "exact",
        observable_strategy: str = "pauli_products",
        device_id: int | None = None,
        network_options: dict | None = None,
        release_workspace: bool = False,
    ) -> None:
        """Initialize the evaluator without importing cuQuantum."""

        super().__init__()

        if max_bond_dim is not None and max_bond_dim <= 0:
            raise ValueError("max_bond_dim must be positive or None.")
        if abs_cutoff is not None and abs_cutoff < 0.0:
            raise ValueError("abs_cutoff must be non-negative or None.")
        if rel_cutoff is not None and rel_cutoff < 0.0:
            raise ValueError("rel_cutoff must be non-negative or None.")
        if precision not in {"fp64", "fp32"}:
            raise ValueError("precision must be either 'fp64' or 'fp32'.")

        if svd_algo is not None:
            svd_algo = svd_algo.lower()
            if svd_algo not in {"gesvd", "gesvdj", "gesvdp", "gesvdr"}:
                raise ValueError("svd_algo must be one of gesvd, gesvdj, gesvdp, or gesvdr.")

        if mpo_application is not None:
            mpo_application = mpo_application.lower()
            if mpo_application not in {"exact", "approximate"}:
                raise ValueError("mpo_application must be 'exact', 'approximate', or None.")

        if observable_strategy not in CUQUANTUM_OBSERVABLE_STRATEGIES:
            raise ValueError(
                "observable_strategy must be one of " f"{sorted(CUQUANTUM_OBSERVABLE_STRATEGIES)}."
            )

        self._max_bond_dim = max_bond_dim
        self._abs_cutoff = abs_cutoff
        self._rel_cutoff = rel_cutoff
        self._precision = precision
        self._dtype = "complex64" if precision == "fp32" else "complex128"
        self._np_dtype = np.complex64 if precision == "fp32" else np.complex128
        self._svd_algo = svd_algo
        self._mpo_application = mpo_application
        self._observable_strategy = observable_strategy
        self._device_id = device_id
        self._network_options = dict(network_options or {})
        self._release_workspace = release_workspace

        self._cuquantum_classes = None
        self._operator = None
        self._operator_cost_op: SparsePauliOp | None = None
        self._energy_offset = 0.0
        self._results_last_iteration = {}

    # pylint: disable=too-many-positional-arguments,arguments-differ
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: list[float],
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the default QAOA energy for the given diagonal cost operator."""

        self._validate_qaoa_features(mixer, initial_state, ansatz_circuit)

        if len(params) % 2 != 0:
            raise KeyError("Number of parameters must be an even integer")

        terms, identity_offset = self._terms_from_cost_operator(cost_op)
        if not terms:
            self._results_last_iteration = self._results_metadata(
                state_norm=1.0,
                identity_offset=identity_offset,
            )
            return identity_offset

        self._ensure_operator(cost_op, terms, identity_offset)

        _, _, NetworkState = self._get_cuquantum_classes()

        state = NetworkState(
            [2 for _ in range(cost_op.num_qubits)],
            dtype=self._dtype,
            config=self._mps_config(),
            options=self._network_options_for_runtime(),
        )
        #norm = state.compute_norm()
        try:
            self._apply_qaoa_state(state, cost_op.num_qubits, terms, params)
            print(self._operator)
            expectation = state.compute_expectation(
                self._operator,
            )
        finally:
            state.free()

        energy = self._as_complex_scalar(expectation)
        energy += self._energy_offset
        energy = float(np.real(energy))
        return energy

    def get_results_from_last_iteration(self) -> dict:
        """Return cuTensorNet configuration metadata from the last evaluation."""

        return self._results_last_iteration

    def to_config(self) -> dict:
        """Json serializable config to keep track of how results are generated."""

        config = super().to_config()
        config.update(
            {
                "max_bond_dim": self._max_bond_dim,
                "abs_cutoff": self._abs_cutoff,
                "rel_cutoff": self._rel_cutoff,
                "precision": self._precision,
                "svd_algo": self._svd_algo,
                "mpo_application": self._mpo_application,
                "observable_strategy": self._observable_strategy,
                "device_id": self._device_id,
                "network_options": self._network_options,
                "release_workspace": self._release_workspace,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "CuQuantumMPSEvaluator":
        """Initialize the evaluator from a configuration dictionary."""

        return cls(**config)

    @classmethod
    def parse_init_kwargs(cls, init_kwargs: str | None = None) -> dict:
        """Parse ``max_bond_abs_cutoff_rel_cutoff_precision_svd_algo`` strings."""

        if init_kwargs is None:
            return dict()

        init_args = init_kwargs.split("_")
        config = {}
        keys = ["max_bond_dim", "abs_cutoff", "rel_cutoff", "precision", "svd_algo"]
        converters = [int, float, float, str, str]

        for idx, (key, converter) in enumerate(zip(keys, converters)):
            if idx < len(init_args) and init_args[idx].lower() != "none":
                config[key] = converter(init_args[idx])

        return config

    def _get_cuquantum_classes(self):
        """Import and cache cuQuantum classes."""

        if self._cuquantum_classes is None:
            self._cuquantum_classes = _require_cuquantum()

        return self._cuquantum_classes

    def _mps_config(self):
        """Create the cuTensorNet MPS configuration."""

        MPSConfig, _, _ = self._get_cuquantum_classes()
        kwargs = {}

        if self._max_bond_dim is not None:
            kwargs["max_extent"] = self._max_bond_dim
        if self._abs_cutoff is not None:
            kwargs["abs_cutoff"] = self._abs_cutoff
        if self._rel_cutoff is not None:
            kwargs["rel_cutoff"] = self._rel_cutoff
        if self._svd_algo is not None:
            kwargs["algorithm"] = self._svd_algo
        if self._mpo_application is not None:
            kwargs["mpo_application"] = self._mpo_application

        return MPSConfig(**kwargs)

    def _network_options_for_runtime(self) -> dict | None:
        """Return cuTensorNet NetworkOptions as a dictionary."""

        options = dict(self._network_options)
        if self._device_id is not None:
            options["device_id"] = self._device_id

        return options or None

    def _results_metadata(self, state_norm: float, identity_offset: float) -> dict:
        """Return metadata for the last energy evaluation."""

        return {
            "state_norm": state_norm,
            "observable_strategy": self._observable_strategy,
            "precision": self._precision,
            "max_bond_dim": self._max_bond_dim,
            "abs_cutoff": self._abs_cutoff,
            "rel_cutoff": self._rel_cutoff,
            "svd_algo": self._svd_algo,
            "mpo_application": self._mpo_application,
            "identity_offset": identity_offset,
        }

    def _ensure_operator(
        self,
        cost_op: SparsePauliOp,
        terms: Sequence[_DiagonalZTerm],
        identity_offset: float,
    ) -> None:
        """Build and cache the cuQuantum observable for the current cost operator."""

        if self._operator is not None and cost_op.equiv(self._operator_cost_op):
            return

        _, NetworkOperator, _ = self._get_cuquantum_classes()
        operator = NetworkOperator(
            [2] * cost_op.num_qubits,
            dtype=self._dtype,
            options=self._network_options_for_runtime(),
        )

        self._append_pauli_product_terms(operator, terms)
        self._energy_offset = identity_offset

        self._operator = operator
        self._operator_cost_op = cost_op.copy()

    def _append_pauli_product_terms(
        self,
        operator,
        terms: Sequence[_DiagonalZTerm],
    ) -> None:
        """Append identity-free Pauli product terms to a cuQuantum NetworkOperator."""

        z_gate = self._z_gate()
        for term in terms:
            operator.append_product(
                term.coefficient,
                [[qubit] for qubit in term.qubits],
                [z_gate for _ in term.qubits],
            )
            

    def _apply_qaoa_state(
        self,
        state,
        n_qubits: int,
        terms: Sequence[_DiagonalZTerm],
        params: Sequence[float],
    ) -> None:
        """Build the QAOA state by applying gates directly to a NetworkState."""

        layer_count = len(params) // 2
        h_gate = self._h_gate()

        for qubit in range(n_qubits):
            state.apply_tensor_operator([qubit], h_gate, unitary=True)

        for layer in range(layer_count):
            gamma = params[layer_count + layer]
            for term in terms:
                if len(term.qubits) == 1:
                    theta = 2.0 * gamma * term.coefficient
                    state.apply_tensor_operator(
                        [term.qubits[0]],
                        self._rz_gate(theta),
                        unitary=True,
                    )
                elif len(term.qubits) == 2:
                    theta = 2.0 * gamma * term.coefficient
                    q0, q1 = sorted(term.qubits)

                    # --- bring qubits together ---
                    # Move q1 left until it is next to q0
                    for k in range(q1 - 1, q0, -1):
                        state.apply_tensor_operator(
                            [k, k + 1],
                            self._swap_gate(),
                            unitary=True,
                        )
                    # --- apply RZZ on adjacent qubits ---
                    state.apply_tensor_operator(
                        [q0, q0 + 1],
                        self._rzz_gate(theta),
                        unitary=True,
                    )

                    # --- undo swaps ---
                    for k in range(q0 + 1, q1):
                        state.apply_tensor_operator(
                            [k, k + 1],
                            self._swap_gate(),
                            unitary=True,
                        )

            beta = params[layer]
            rx_gate = self._rx_gate(2.0 * beta)
            for qubit in range(n_qubits):
                state.apply_tensor_operator([qubit], rx_gate, unitary=True)

    @staticmethod
    def _terms_from_cost_operator(cost_op: SparsePauliOp) -> tuple[list[_DiagonalZTerm], float]:
        """Extract supported diagonal terms and the identity offset from a cost operator."""

        terms = []
        identity_offset = 0.0

        for pauli_str, coefficient in cost_op.to_list():
            coefficient = complex(coefficient)
            if abs(coefficient.imag) > 1.0e-12:
                raise NotImplementedError("Complex Hamiltonian coefficients are not supported.")
            if any(char not in {"I", "Z"} for char in pauli_str):
                raise NotImplementedError(
                    "CuQuantumMPSEvaluator supports only diagonal I/Z cost operators."
                )

            qubits = tuple(idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z")
            if len(qubits) == 0:
                identity_offset += float(coefficient.real)
            elif len(qubits) <= 2:
                terms.append(_DiagonalZTerm(qubits=qubits, coefficient=float(coefficient.real)))
            else:
                raise NotImplementedError(
                    "CuQuantumMPSEvaluator currently supports only one- and two-local Z terms."
                )

        return terms, identity_offset

    @staticmethod
    def _validate_qaoa_features(
        mixer: QuantumCircuit | None,
        initial_state: QuantumCircuit | None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None,
    ) -> None:
        """Reject features outside the first cuQuantum MPS implementation."""

        if mixer is not None:
            raise NotImplementedError("Custom mixers are not supported; use the default X mixer.")
        if initial_state is not None:
            raise NotImplementedError("Custom initial states are not supported.")
        if ansatz_circuit is not None:
            raise NotImplementedError("Custom ansatz circuits are not supported.")

    def _h_gate(self) -> np.ndarray:
        """Return the Hadamard gate."""

        return np.array(
            [[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)]],
            dtype=self._np_dtype,
        )

    def _z_gate(self) -> np.ndarray:
        """Return the Pauli-Z gate."""

        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=self._np_dtype)

    def _rx_gate(self, theta: float) -> np.ndarray:
        """Return the RX(theta) gate."""

        return np.array(
            [
                [np.cos(theta / 2.0), -1.0j * np.sin(theta / 2.0)],
                [-1.0j * np.sin(theta / 2.0), np.cos(theta / 2.0)],
            ],
            dtype=self._np_dtype,
        )

    def _rz_gate(self, theta: float) -> np.ndarray:
        """Return the RZ(theta) gate."""

        return np.array(
            [
                [np.exp(-0.5j * theta), 0.0],
                [0.0, np.exp(0.5j * theta)],
            ],
            dtype=self._np_dtype,
        )

    def _rzz_gate(self, theta: float) -> np.ndarray:
        """Return the RZZ(theta) gate in cuQuantum tensor-operator axis order."""

        # Diagonal entries
        diag = np.array(
            [
                np.exp(-0.5j * theta),
                np.exp(0.5j * theta),
                np.exp(0.5j * theta),
                np.exp(-0.5j * theta),
            ],
            dtype=self._np_dtype,
        )

        # Build operator explicitly in (out0, out1, in0, in1)
        gate = np.zeros((2, 2, 2, 2), dtype=self._np_dtype)

        for i in range(2):
            for j in range(2):
                idx = 2 * i + j
                gate[i, j, i, j] = diag[idx]

        return gate
    def _swap_gate(self):
        swap = np.zeros((2, 2, 2, 2), dtype=self._np_dtype)
        for i in range(2):
            for j in range(2):
                swap[j, i, i, j] = 1.0
        return swap

    @staticmethod
    def _as_complex_scalar(value) -> complex:
        """Convert NumPy/CuPy/Torch scalar-like values to a Python complex."""

        if hasattr(value, "get"):
            value = value.get()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "item"):
            value = value.item()
        return complex(value)
