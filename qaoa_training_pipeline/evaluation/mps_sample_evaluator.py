#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sample-based MPS evaluator"""

import time
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator


class SampleEvaluator(BaseEvaluator):
    """Approximate the energy by sampling from a MPS.
    
    This MPS-based energy evaluator does not contract the MPS to compute the value of
    an observable. Instead, we draw samples `x` from the MPS and then evaluate the energy
    of the observable for each sample. 
    """

    def __init__(self, chi=None, max_parallel_threads=None, shots=None):
        """Initialize the class."""
        super().__init__()

        self._cost_op = None
        self._counts = []

        self.chi = chi or 20
        self.max_parallel_threads = max_parallel_threads or 10

        self._backend = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=self.chi,
            max_parallel_threads=self.max_parallel_threads,
        )
        self._backend.options.use_fractional_gates = False

        self._sampler = BackendSamplerV2(backend=self._backend)

        self._shots = shots or 1000
        self.energies = []

    @property
    def cost_op(self):
        """Returns the cost operator"""
        return self._cost_op

    @cost_op.setter
    def cost_op(self, cost_op):
        """Set the cost operator.
        
        This property setter computes some internal variables that help speed-up the computation
        of the energy for each sample `x`.
        """
        self._cost_op = cost_op
        self._reals = []
        self._ainds = []
        start = time.time()
        for pauli in self._cost_op:
            indices = tuple(idx for idx, val in enumerate(pauli.paulis[0].z) if val)
            self._ainds.append(indices)
            self._reals.append(np.real(pauli.coeffs[0]))

        self._init_time = time.time() - start

    def energy(self, sample: str) -> float:
        """Computes the energy for a given sample"""
        sample = [val == "1" for val in sample[::-1]]

        energy = 0
        for aidx, val in enumerate(self._reals):
            if len(self._ainds[aidx]) == 1:
                if sample[self._ainds[aidx][0]]:
                    energy -= val
                else:
                    energy += val

            if len(self._ainds[aidx]) == 2:
                if sample[self._ainds[aidx][0]] == sample[self._ainds[aidx][1]]:
                    energy += val
                else:
                    energy -= val
            if len(self._ainds[aidx]) == 0:
                energy += 1

        return energy

    def total_energy(self, counts: dict) -> float:
        """Compute the energy of the counts."""
        tot_energy = 0
        self.energies = []
        shots = sum(counts.values())
        for sample, count in counts.items():
            self.energies.append(self.energy(sample))
            tot_energy += self.energies[-1] * count / shots

        return tot_energy

    # pylint: disable=too-many-positional-arguments
    def evaluate(self, cost_op, params, mixer=None, initial_state=None, ansatz_circuit=None):
        """Evaluate the energy."""

        if isinstance(ansatz_circuit, SparsePauliOp):
            ansatz_op = ansatz_circuit
        elif ansatz_circuit is None:
            ansatz_op = cost_op
        else:
            raise NotImplementedError(
                "Custom ansatz circuits in format"
                f"{ansatz_circuit.__class__.__name__} are not yet supported."
            )
        # Set the cost op. We do not validate that the existing cost op,
        # if present, is the same as the given cost op.
        if self._cost_op is None:
            self.cost_op = ansatz_op

        ansatz = qaoa_ansatz(
            cost_operator=ansatz_op,
            reps=len(params) // 2,
        ).decompose()

        ansatz.measure_all()

        pub = (ansatz, params, self._shots)
        result = self._sampler.run([pub]).result()

        self._counts = result[0].data.meas.get_counts()

        return self.total_energy(self._counts)

    def get_results_from_last_iteration(self):
        """Return the results from the last iteration."""
        return {"counts": self._counts}

    def to_config(self):
        config = super().to_config()
        config["chi"] = (self.chi,)
        config["max_parallel_threads"] = (self.max_parallel_threads,)

        return config

    def cvar(self, energies: list, alpha=1.00) -> float:
        """Compute the CVaR for given energies."""
        sorted_energies = sorted(energies)
        end_idx = max(int(alpha * len(energies)), 1)

        return float(np.sum(sorted_energies[0:end_idx]) / end_idx)
