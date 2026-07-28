#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
import quimb.tensor as qtn
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
    """Inherits from QAOACircuitMPSRepresentation and applies the same logic, but recompresses
    to a fixed bond dimension every `gates_per_block` individual two-qubit gate applications.
    """

    # pylint: disable=too-many-positional-arguments
    def configure_compression(self, max_bond, gates_per_block, method, site_tags, compress_opts):
        self._compress_max_bond = max_bond
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
                # Plain SVD-based block compression -- no ALS-specific options apply here.
                psi = qtn.tensor_network_1d_compress(
                    self.get_underlying_tn(),
                    max_bond=self._compress_max_bond,
                    method="dm",
                    site_tags=self._site_tags,
                    **self._compress_opts,
                )
            else:
                # "fit" -> warm-started (zip-up/TEBD-like guess) single-site ALS sweep, matching
                # the paper's recommended initialization strategy discussed above. `cutoff` is
                # for building that initial guess; `cutoff_fit` (0, i.e. no truncation) is for
                # the ALS sweep itself, which is only meaningful with method="fit-zipup" (the
                # plain "fit" method does not accept `cutoff_fit` at all).
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

            # `_apply_two_qubit_gate` (the parent class) assumes the MPS it's handed has the
            # same index-naming/axis conventions a natively-built CircuitMPS has (e.g. it looks
            # up bonds positionally via `psi.inner_inds()[i]`). tensor_network_1d_compress's
            # output does not reliably satisfy this -- its inner_inds() come back in an
            # unrelated order, and even after fixing that ordering (via the position-based
            # `.bond(i, i+1)`, not insertion-order-based `inner_inds()`), the parent's internal
            # `shift_orthogonality_center` call can still misbehave on it, causing a *different*
            # tensor/bond to be produced than what `_apply_two_qubit_gate` expects -- surfacing
            # as a `KeyError` on `info[("singular_values", ...)]` a gate or more later. The
            # robust fix is to not reuse tensor_network_1d_compress's internal representation at
            # all: extract the raw site arrays (in the correct leg order) and rebuild a brand
            # new `MatrixProductState` from scratch, which goes through the same construction
            # path as any natively-built MPS and so is guaranteed to satisfy those assumptions.
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
        self, max_bond, gates_per_block=1, method="fit", swap_strategy=None, **compress_opts
    ):
        self.max_bond = max_bond
        self.gates_per_block = gates_per_block
        self.method = method
        self.swap_strategy = swap_strategy
        self.compress_opts = compress_opts

    def evaluate(self, cost_op, params) -> float:
        n_qubits = cost_op.num_qubits
        edges = operator_to_list_of_hyper_edges(cost_op)
        if any(len(edge[0]) != 2 for edge in edges):
            raise NotImplementedError("Prototype only supports quadratic (edge) cost operators.")

        depth = len(params) // 2
        betas, gammas = params[:depth], params[depth:]

        swap_strategy = self.swap_strategy or make_swap_strategy(
            [tuple(edge[0]) for edge in edges], n_qubits
        )

        circuit = _BlockCompressingCircuit.construct_from_list_of_edges(
            edges,
            truncation_threshold=None,
            max_bond_dim=None,
            swap_strategy=swap_strategy,
        )
        site_tags = tuple(f"I{i}" for i in range(n_qubits))
        circuit.configure_compression(
            self.max_bond, self.gates_per_block, self.method, site_tags, self.compress_opts
        )
        circuit._apply_initial_layer()

        rep = 1
        for layer in range(depth):
            circuit._apply_layer_ansatz_swap_strat(gammas[layer], rep)
            circuit._apply_mixing_layer(betas[layer])
            rep += 1

        # Same net-permutation correction MPSEvaluator applies for use_swap_strategy=True.
        eval_cost_op = cost_op
        if depth % 2 == 1:
            inv_perm = swap_strategy.inverse_composed_permutation(len(swap_strategy))
            permutation = [inv_perm.index(idx) for idx in range(len(inv_perm))]
            eval_cost_op = cost_op.apply_layout(permutation)

        cost_function = QAOACostFunction(eval_cost_op)
        mpo = cost_function.mpo.mpo
        psi = circuit.get_underlying_tn()
        psi_dagger = psi.H
        psi_dagger.reindex_(dict(zip(mpo.lower_inds, mpo.upper_inds)))
        network = psi_dagger & mpo & psi
        energy = network.contract(..., optimize="auto-hq") / psi.norm() ** 2
        return float(np.real(energy))
