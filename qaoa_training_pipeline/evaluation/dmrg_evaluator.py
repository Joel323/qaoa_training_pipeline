#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class implementing a DMRG-style MPS"""

import numpy as np
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from quimb.tensor import CircuitMPS

from qaoa_training_pipeline.utils.graph_utils import (
    make_swap_strategy,
    operator_to_list_of_hyper_edges,
)
from qaoa_training_pipeline.utils.tns_utils.qaoa_circuit_mps import (
    QAOACircuitMPSRepresentation,
)
from qaoa_training_pipeline.utils.tns_utils.qaoa_cost_function import QAOACostFunction


class _BlockCompressingCircuit(QAOACircuitMPSRepresentation):
    """Same as QAOACircuitMPSRepresentation, but recompresses to a fixed bond dimension every
    `gates_per_block` individual two-qubit gate applications (RZZ or SWAP), instead of only
    truncating (or not) at the end of a whole QAOA layer.
    """

    def __init__(
        self,
        edges,
        growth_max_bond,
        swap_strategy,
        site_tags: tuple[str, ...],
        compress_max_bond: int = 4,
        gates_per_block: int = 280,
        method: str = "fit",
        truncation_threshold=None,
        store_intermediate_schmidt_values=False,
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        **compress_opts,
    ):
        """Configure internal variables controlling the compression of the MPS state"""
        n_qubits = max(max(i[0]) for i in edges) + 1
        adjacency_matrix = np.zeros((n_qubits, n_qubits), dtype=float)
        list_of_higher_order_terms = []

        # Loop over the edges and classify them
        for i_edge in edges:
            # First-order terms are in the diagonal of the adjacency matrix
            if len(i_edge[0]) == 1:
                adjacency_matrix[i_edge[0][0], i_edge[0][0]] = i_edge[1]
            # Second-order edges
            elif len(i_edge[0]) == 2:
                adjacency_matrix[i_edge[0][0], i_edge[0][1]] = i_edge[1]
                adjacency_matrix[i_edge[0][1], i_edge[0][0]] = i_edge[1]
            else:
                list_of_higher_order_terms.append(i_edge)

        super().__init__(
            n_qubits,
            adjacency_matrix,
            truncation_threshold,
            growth_max_bond,
            swap_strategy=swap_strategy,
            list_of_hyperedges=list_of_higher_order_terms,
            mixer=mixer,
            initial_state=initial_state,
            store_intermediate_schmidt_values=store_intermediate_schmidt_values,
        )
        self._compress_max_bond = compress_max_bond
        self._compress_block = gates_per_block
        self._compress_method = method
        self._site_tags = site_tags
        self._compress_opts = compress_opts
        self._gate_count = 0

    def _apply_two_qubit_gate(self, i_qubit, array, absorb_left):
        result = super()._apply_two_qubit_gate(i_qubit, array, absorb_left)
        self._gate_count += 1
        if self._gate_count % self._compress_block == 0:
            if self._compress_method == "dm":
                psi = qtn.tensor_network_1d_compress(
                    self.get_underlying_tn(),
                    max_bond=self._compress_max_bond,
                    method="dm",
                    site_tags=self._site_tags,
                    **self._compress_opts,
                )
            elif self._compress_method == "fit-zipup":
                psi = qtn.tensor_network_1d_compress(
                    self.get_underlying_tn(),
                    max_bond=self._compress_max_bond,
                    method="fit-zipup",
                    initial_bond_dim=self._compress_max_bond,
                    bsz=1,
                    cutoff=1e-10,
                    cutoff_fit=0.0,
                    max_iterations=4,
                    tol=1e-8,
                    normalize=True,
                    site_tags=self._site_tags,
                    **self._compress_opts,
                )
            else:
                raise ValueError(f"{self._compress_method} is not supported")

            arrays = []
            for i, tag in enumerate(self._site_tags):
                tensor = psi[tag]
                if i == 0:
                    order = (f"k{i}", psi.bond(i, i + 1))
                elif i == self.n_qubits - 1:
                    order = (psi.bond(i - 1, i), f"k{i}")
                else:
                    order = (psi.bond(i - 1, i), f"k{i}", psi.bond(i, i + 1))
                arrays.append(tensor.transpose(*order).data)
            fresh_mps = qtn.MatrixProductState(arrays, shape="lpr")
            fresh_mps.canonize(0)

            self._mps_representation = CircuitMPS(self.n_qubits, psi0=fresh_mps)
            self._canonization_center = 0
        return result


class DMRGStyleEvaluator:
    """Prototype of the DMRG-style circuit-compression algorithm of arXiv:2207.05612.

    Drives `QAOACircuitMPSRepresentation` (with truncation disabled) via its SWAP-strategy
    machinery, and compresses back to `max_bond` every `gates_per_block` individual two-qubit
    gates via quimb's `tensor_network_1d_compress` (`method="fit"` = single-site ALS / DMRG-style
    sweep, warm-started from a zip-up guess; `method="dm"` = direct SVD-based block compression).
    """

    def __init__(
        self,
        compress_max_bond,
        growth_max_bond=None,
        gates_per_block=1,
        method="fit-zipup",
        swap_strategy=None,
        **compress_opts,
    ):
        self.compress_max_bond = compress_max_bond
        self.growth_max_bond = growth_max_bond
        self.gates_per_block = gates_per_block
        self.method = method
        self.compress_opts = compress_opts
        self.swap_strategy = swap_strategy

    def evaluate(self, cost_op, params) -> float:
        """Provides energy evaluation coming from contraction of the MPS obtained by using the DMRG-style algorithm"""
        n_qubits = cost_op.num_qubits
        edges = operator_to_list_of_hyper_edges(cost_op)
        if any(len(edge[0]) != 2 for edge in edges):
            raise NotImplementedError("Prototype only supports quadratic (edge) cost operators.")

        self.swap_strategy = self.swap_strategy or make_swap_strategy(
            [tuple(edge[0]) for edge in edges], n_qubits
        )
        site_tags = tuple(f"I{i}" for i in range(n_qubits))
        circuit = _BlockCompressingCircuit(
            edges=edges,
            growth_max_bond=self.growth_max_bond,
            swap_strategy=self.swap_strategy,
            site_tags=site_tags,
            compress_max_bond=self.compress_max_bond,
            gates_per_block=self.gates_per_block,
            method=self.method,
            truncation_threshold=None,
            store_intermediate_schmidt_values=False,
        )

        depth = len(params) // 2
        betas, gammas = params[:depth], params[depth:]

        circuit._apply_initial_layer()

        rep = 1
        for layer in range(depth):
            circuit._apply_layer_ansatz_swap_strat(gammas[layer], rep)
            circuit._apply_mixing_layer(betas[layer])
            rep += 1

        # Same net-permutation correction MPSEvaluator applies for use_swap_strategy=True.
        eval_cost_op = cost_op
        if depth % 2 == 1:
            inv_perm = self.swap_strategy.inverse_composed_permutation(len(self.swap_strategy))
            permutation = [inv_perm.index(idx) for idx in range(len(inv_perm))]
            eval_cost_op = cost_op.apply_layout(permutation)

        cost_function = QAOACostFunction(eval_cost_op)
        mpo = cost_function.mpo.mpo
        psi = circuit.get_underlying_tn()
        psi_dagger = psi.H
        assert mpo.lower_inds is not None and mpo.upper_inds is not None
        psi_dagger.reindex_(dict(zip(mpo.lower_inds, mpo.upper_inds)))
        network = psi_dagger & mpo & psi
        raw = network.contract(tags=..., optimize="auto-hq")
        assert isinstance(raw, (float, complex))
        norm = psi.norm()
        assert isinstance(norm, (float, complex))
        energy = raw / norm**2
        return float(np.real(energy))
