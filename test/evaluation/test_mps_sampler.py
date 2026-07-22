#
#
# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit Aer MPS-based QAOA evaluator tests."""

from unittest import TestCase

from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.mps_sample_evaluator import SampleEvaluator


class TestSampleEvaluator(TestCase):
    """Test that the MPS evaluator from qiskit aer works."""

    def setUp(self):
        """Setup the variables."""
        self.cost_op = SparsePauliOp.from_list([("II", 1.0), ("IZ", 1.0), ("ZZ", 1.0)])
        self.evaluator = SampleEvaluator(shots=10000, chi=32)

    def qiskit_circuit_simulation(self, cost_op, params):
        """This is the baseline simulation based on Qiskit."""

        ansatz = qaoa_ansatz(cost_op, reps=len(params) // 2)
        estimator = StatevectorEstimator()
        ansatz.assign_parameters(params, inplace=True)
        result = estimator.run([(ansatz, cost_op, [])]).result()
        return float(result[0].data.evs)

    def test_evaluate(self):
        """Basic test of the evaluator."""
        angles = [0.1, 0.3]
        energy1 = self.evaluator.evaluate(self.cost_op, params=angles)
        energy2 = self.qiskit_circuit_simulation(self.cost_op, angles)
        self.assertTrue(abs(energy1 - energy2) < 0.02)

    def test_custom_ansatz(self):
        """Test that we can construct the ansatz from a different operator."""
        ansatz_op = SparsePauliOp.from_list([("ZI", 1)])

        angles = [1.2, 1.3]

        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=ansatz_op)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles)

        self.assertTrue(abs(energy1 - energy2) > 0.02)

        energy1 = self.evaluator.evaluate(self.cost_op, params=angles, ansatz_circuit=self.cost_op)
        energy2 = self.evaluator.evaluate(self.cost_op, params=angles)

        self.assertTrue(abs(energy1 - energy2) < 0.02)
