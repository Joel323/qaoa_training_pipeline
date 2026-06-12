# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pyrauli-based QAOA evaluator using Pauli propagation."""

import importlib.util
import pyrauli  
from pyrauli import from_qiskit

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp

from qaoa_training_pipeline.evaluation.base_evaluator import BaseEvaluator

# cspell: ignore pyrauli qarg qargs eiht
# Safely import pyrauli if it is installed.

pyrauli_loader = importlib.util.find_spec("pyrauli")
HAS_PYRAULI = pyrauli_loader is not None
if HAS_PYRAULI:

    # Supported gates based on pyrauli API
    # No need to restrict basis gates since pyrauli.from_qiskit() handles conversion
    SUPPORTED_GATES = None
else:
    SUPPORTED_GATES = []
    # pylint: disable=invalid-name
    pyrauli = None  # type: ignore


class PyRauliEvaluator(BaseEvaluator):
    """Evaluator based on the pyrauli Pauli propagation method.

    This class requires that the system has the pyrauli library installed.
    Pyrauli is a Python library for efficient Pauli propagation that can be
    installed via pip: `pip install pyrauli`.
    """

    def __init__(self, pyrauli_kwargs: dict | None = None):
        """Initialize the pyrauli evaluator.

        Args:
            pyrauli_kwargs: Keyword arguments for pyrauli truncation configuration.
                The most relevant parameters are:

                 - `threshold`: Coefficient threshold below which terms are truncated
                     during Pauli propagation. Default is 1e-6.
                 - `max_terms`: Maximum number of terms to keep. Default is None (no limit).

                If None is given then we default to `threshold=1e-6` and `max_terms=None`.
        """

        # Check if pyrauli is available
        if not HAS_PYRAULI:
            raise ImportError(
                f"{self.__class__.__name__} requires the pyrauli library. "
                f"Please install it using: pip install pyrauli"
            )

        # Default configuration for truncation
        self.pyrauli_kwargs = {
            "threshold": 1e-6,
            "max_terms": None,
        }
        if pyrauli_kwargs is not None:
            self.pyrauli_kwargs.update(pyrauli_kwargs)

    # pylint: disable=too-many-positional-arguments
    def evaluate(
        self,
        cost_op: SparsePauliOp,
        params: list[float],
        mixer: QuantumCircuit | None = None,
        initial_state: QuantumCircuit | None = None,
        ansatz_circuit: QuantumCircuit | SparsePauliOp | None = None,
    ) -> float:
        """Evaluate the QAOA circuit parameters using pyrauli.

        Args:
            cost_op: The cost operator that defines the cost Hamiltonian.
            params: The parameters for QAOA. The length of this sequence will
                determine the depth of the QAOA.
            mixer: The mixer operator. Defaults to None.
            initial_state: The initial state circuit. Defaults to None.
            ansatz_circuit: The ansatz circuit for the cost operator. Defaults to None.

        Returns:
            The expectation value as a float.
        """

        if ansatz_circuit is None:
            ansatz_circuit = cost_op
        else:
            if not isinstance(ansatz_circuit, SparsePauliOp):
                raise NotImplementedError(
                    "Only ansatz_circuit specified by a sparse Pauli operator is supported."
                )

        # Build QAOA circuit
        circuit = qaoa_ansatz(
            ansatz_circuit,
            reps=len(params) // 2,
            initial_state=initial_state,
            mixer_operator=mixer,  # type: ignore
        )
        bound_circuit = circuit.assign_parameters(params, inplace=False)
        
        # Use the bound circuit directly - pyrauli.from_qiskit() handles gate conversion
        circuit = bound_circuit

        # Convert to pyrauli format
        pyrauli_circuit = from_qiskit(circuit)
        pyrauli_observable = self.sparse_pauli_op_to_pyrauli(cost_op)

        # Set up truncation if specified
        if self.pyrauli_kwargs["threshold"] is not None:
            assert pyrauli, "pyrauli must be defined before calling evaluate()"
            threshold = self.pyrauli_kwargs["threshold"]
            truncator = pyrauli.CoefficientTruncator(threshold)
            policy = pyrauli.AlwaysAfterSplittingPolicy()
            pyrauli_circuit.set_truncator(truncator)
            pyrauli_circuit.set_truncate_policy(policy)

        # Compute expectation value using pyrauli
        # Circuit.expectation_value() propagates the observable through the circuit
        # and computes ⟨ψ|O|ψ⟩ where |ψ⟩ is the output state
        expectation_value = pyrauli_circuit.run(pyrauli_observable).expectation_value()
        
        return float(expectation_value)

    def sparse_pauli_op_to_pyrauli(self, op: SparsePauliOp):
        """Convert a Qiskit SparsePauliOp to a pyrauli Observable.

        Args:
            op: The Qiskit SparsePauliOp to convert.

        Returns:
            A pyrauli Observable representing the operator.
        """
        if not HAS_PYRAULI or pyrauli is None:
            raise RuntimeError(
                "pyrauli is not available. Install it using: pip install pyrauli"
            )

        n_qubits = op.num_qubits
        
        # Build list of PauliTerm objects with coefficients
        pauli_terms = []
        
        for pauli_str, coeff in zip(op.paulis.to_labels(), op.coeffs):
            # IMPORTANT: PyRauli uses reversed qubit ordering compared to Qiskit
            # Qiskit 'ZI' means Z on qubit 0, I on qubit 1
            # PyRauli 'IZ' means Z on qubit 0, I on qubit 1
            # So we need to reverse the Pauli string
            reversed_pauli_str = pauli_str[::-1]
            
            # Create PauliTerm with coefficient
            # Convert coefficient to float to avoid numpy scalar type issues
            assert pyrauli, "pyrauli must be defined"
            pauli_term = pyrauli.PauliTerm(reversed_pauli_str, float(coeff.real))
            pauli_terms.append(pauli_term)
        
        if pauli_terms:
            # Create Observable from list of PauliTerms
            return pyrauli.Observable(pauli_terms)
        else:
            # Return identity observable if no terms
            assert pyrauli, "pyrauli must be defined"
            # type: ignore is needed because type checker doesn't know num_qubits is always int
            identity_string = "I" * n_qubits  # type: ignore
            return pyrauli.Observable(identity_string, 0.0)

    def qc_to_pyrauli(self, circuit: QuantumCircuit):
        """Convert a Qiskit QuantumCircuit to a pyrauli Circuit.

        Args:
            circuit: The Qiskit circuit with no free parameters.

        Returns:
            A pyrauli Circuit object.

        Raises:
            ValueError: If the circuit has unassigned parameters.
        """
        if len(circuit.parameters) > 0:
            raise ValueError("The provided quantum circuit has unassigned parameters.")

        if not HAS_PYRAULI or pyrauli is None:
            raise RuntimeError(
                "pyrauli is not available. Install it using: pip install pyrauli"
            )

        assert pyrauli, "pyrauli must be defined before calling qc_to_pyrauli()"
        
        # Use pyrauli's built-in Qiskit conversion function
        return pyrauli.from_qiskit(circuit)

    def to_config(self) -> dict:
        """Json serializable config to keep track of how results are generated.

        Returns:
            A dictionary containing the evaluator configuration.
        """
        config = super().to_config()
        config["pyrauli_kwargs"] = self.pyrauli_kwargs
        return config

    @classmethod
    def from_config(cls, config: dict) -> "PyRauliEvaluator":
        """Initialize the pyrauli evaluator from a config dictionary.

        Args:
            config: Configuration dictionary containing pyrauli_kwargs.

        Returns:
            A PyRauliEvaluator instance.
        """
        return cls(**config)

    @classmethod
    # pylint: disable=unused-argument
    def parse_init_kwargs(cls, init_kwargs: str | None = None) -> dict:
        """Parse initialization kwargs from string format.

        Args:
            init_kwargs: String in format "key1:val1:key2:val2:..." or None.

        Returns:
            Dictionary with parsed kwargs suitable for __init__.

        Raises:
            ValueError: If the string format is malformed.
        """
        if init_kwargs is None:
            return dict()

        items = init_kwargs.split(":")

        if len(items) % 2 != 0:
            raise ValueError(
                f"Malformed keyword arguments {init_kwargs}: should be k1:v1:k2:v2_...."
            )

        # Parse key-value pairs and convert to appropriate types
        parsed_kwargs = {}
        for idx in range(0, len(items), 2):
            key = items[idx]
            value_str = items[idx + 1]
            
            # Try to convert to float, otherwise keep as string
            try:
                value = float(value_str)
            except ValueError:
                # Handle special cases like None
                if value_str.lower() == "none":
                    value = None
                else:
                    value = value_str
            
            parsed_kwargs[key] = value

        return {"pyrauli_kwargs": parsed_kwargs}