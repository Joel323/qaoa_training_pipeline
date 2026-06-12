#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pyrauli-based QAOA evaluator tests."""
import numpy as np
import pytest
from test import TrainingPipelineTestCase

from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.pauli_propagation_pyrauli import (
    PyRauliEvaluator
)
from qaoa_training_pipeline.evaluation.statevector_evaluator import (
    StatevectorEvaluator
)


class TestPyRauliEvaluator(TrainingPipelineTestCase):
    """Test that the PyRauli evaluator works."""

    def setUp(self):
        """Setup the evaluators."""
        self.sv_evaluator = StatevectorEvaluator()

    def test_instantiation(self):
        """Test that the evaluator can be instantiated."""
        pyrauli_kwargs = {"threshold": 1e-5}
        evaluator = PyRauliEvaluator(pyrauli_kwargs)
        self.assertIsNotNone(evaluator)
        self.assertEqual(evaluator.pyrauli_kwargs["threshold"], 1e-5)

    def test_config(self):
        """Test configuration serialization."""
        pyrauli_kwargs = {"threshold": 1e-5}
        evaluator = PyRauliEvaluator(pyrauli_kwargs)
        config = evaluator.to_config()
        self.assertIn("pyrauli_kwargs", config)
        self.assertEqual(config["pyrauli_kwargs"]["threshold"], 1e-5)
        
        # Test from_config - wrap pyrauli_kwargs properly
        evaluator2 = PyRauliEvaluator.from_config({"pyrauli_kwargs": config["pyrauli_kwargs"]})
        self.assertEqual(evaluator2.pyrauli_kwargs["threshold"], 1e-5)

    @pytest.mark.timeout(30)
    def test_evaluate_single_pauli(self):
        """Test the evaluate method with a single Pauli term."""
        # Try with just one Pauli term first
        cost_op = SparsePauliOp.from_list([("ZI", 1.0)])
        params = [0.1, 0.4]
        pyrauli_kwargs = {"threshold": 1e-5}
        evaluator = PyRauliEvaluator(pyrauli_kwargs)
        
        # This should work if pyrauli is properly installed
        result = evaluator.evaluate(cost_op, params=params)
        
        # Check that we get a float result
        self.assertIsInstance(result, float)
        # Check that the result is finite
        self.assertTrue(np.isfinite(result))
        
        # Compare with statevector simulation
        sv_result = self.sv_evaluator.evaluate(cost_op, params=params)
        self.assertAlmostEqual(result, sv_result, places=6,
                               msg=f"PyRauli result {result} differs from Statevector result {sv_result}")

    @pytest.mark.timeout(30)
    def test_evaluate_multiple_paulis(self):
        """Test the evaluate method with multiple Pauli terms."""
        cost_op = SparsePauliOp.from_list([("IZ", 0.5), ("ZI", -0.5)])
        params = [0.1, 0.4]
        pyrauli_kwargs = {"threshold": 1e-5}
        evaluator = PyRauliEvaluator(pyrauli_kwargs)
        
        # This should work if pyrauli is properly installed
        result = evaluator.evaluate(cost_op, params=params)
        
        # Check that we get a float result
        self.assertIsInstance(result, float)
        # Check that the result is finite
        self.assertTrue(np.isfinite(result))
        
        # Compare with statevector simulation
        sv_result = self.sv_evaluator.evaluate(cost_op, params=params)
        self.assertAlmostEqual(result, sv_result, places=6,
                               msg=f"PyRauli result {result} differs from Statevector result {sv_result}")


    # def test_optimize(self):
    #     """Data-driven test of optimization."""
    #     trainer = ScipyTrainer(self.evaluator, {"options": {"maxiter": 3, "rhobeg": 0.2}})
    #     result = trainer.train(cost_op=self.cost_op, params0=self.params)
    #     self.assertGreaterEqual(len(result["energy_history"]), 3)
    