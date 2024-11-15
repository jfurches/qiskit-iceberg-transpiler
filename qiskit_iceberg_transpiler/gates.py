r"""A set of modules and classes for working with the [[k+2, k, 2]] iceberg code.
We follow the implementation by Quantinuum [1] to tailor this to trapped-ion processors.

[1] C. Self, M. Benedetti, and D. Amaro, 2024. https://arxiv.org/abs/2211.06703
"""

from qiskit import ClassicalRegister
from qiskit.circuit import Instruction, QuantumCircuit, QuantumRegister, Qubit


class Initialization(Instruction):
    """Implements logical state initialization of the [[k+2, k, 2]] iceberg code"""

    def __init__(self, logical_qubits: int):
        super().__init__(
            "Initialization",
            num_qubits=logical_qubits + 4,
            num_clbits=2,
            params=[logical_qubits],
        )
        self.k = logical_qubits

    def _define(self):
        # This is figure 1c in [1]
        k = QuantumRegister(self.k, "k")  # logical qubit register
        t = QuantumRegister(1, "t")
        b = QuantumRegister(1, "b")
        a = QuantumRegister(2, "a")  # ancillas for readout
        cl_a = ClassicalRegister(2, "cl_a")
        qc = QuantumCircuit(k, t, b, a, cl_a)

        qubits = [t[0], *k, b[0]]

        # Encode the k+2 qubit cat state on qubits [k] ∪ {t, b}
        qc.h(t[0])
        for i in range(len(qubits) - 1):
            qc.cx(qubits[i], qubits[i + 1])

        # Readout any state prep error on our ancillas from t and b
        qc.barrier()
        qc.cx(t[0], a[0])
        qc.cx(b[0], a[0])
        qc.measure(a[0], cl_a[0])

        self._definition = qc


class Syndrome(Instruction):
    """Placeholder for performing syndrome checking.

    Use this if you want to manually add syndrome detection in your circuit. Note this gate is not implementable, the transpiler will replace it with the proper gate.
    """

    def __init__(self, logical_qubits: int):
        """Construct a placeholder for syndrome checking on k logical qubits.

        This should be applied just on the logical qubits.
        """
        super().__init__(
            "Syndrome",
            num_qubits=logical_qubits,
            num_clbits=0,
            params=[logical_qubits],
        )
        self.k = logical_qubits

    def _define(self):
        raise NotImplementedError("This should not be synthesized directly")


class SyndromeMeasurement(Instruction):
    """Implements syndrome checking for the [[k+2, k, 2]] iceberg code"""

    def __init__(self, logical_qubits: int):
        """Constructs the syndrome detection circuit

        Args:
            logical_qubits: The number of logical qubits in the code
        """
        super().__init__(
            "SyndromeMeasurement",
            num_qubits=logical_qubits + 4,
            num_clbits=2,
            params=[logical_qubits],
        )
        self.k = logical_qubits

    def _define(self):
        # This is figure 1d in [1]
        k = QuantumRegister(self.k, "k")  # logical qubit register
        t = QuantumRegister(1, "t")
        b = QuantumRegister(1, "b")
        a = QuantumRegister(2, "a")  # ancillas for readout
        cl_a = ClassicalRegister(2, "cl_a")
        qc = QuantumCircuit(k, t, b, a, cl_a)

        qubits = [t[0], *k, b[0]]

        # Mid circuit reset ancillas
        qc.reset(a)
        qc.h(a[1])

        # Construct syndrome measurement following ABB...BA pattern
        for idx in range(0, len(qubits), 2):
            if idx in {0, len(qubits) - 2}:
                self.pattern_A(qc, a, qubits[idx], qubits[idx + 1])
            else:
                self.pattern_B(qc, a, qubits[idx], qubits[idx + 1])

        qc.h(a[1])
        qc.measure(a[0], cl_a[0])
        qc.measure(a[1], cl_a[1])

        self._definition = qc

    def pattern_A(self, qc: QuantumCircuit, a: QuantumRegister, q1: Qubit, q2: Qubit):
        qc.cx(a[1], q1)
        qc.cx(q1, a[0])
        qc.cx(q2, a[0])
        qc.cx(a[1], q2)

    def pattern_B(self, qc: QuantumCircuit, a: QuantumRegister, q1: Qubit, q2: Qubit):
        qc.cx(a[1], q1)
        qc.cx(q1, a[0])
        qc.cx(a[1], q2)
        qc.cx(q2, a[0])


class LogicalMeasurement(Instruction):
    """Logical measurement layer for the iceberg code"""

    def __init__(self, logical_qubits: int):
        super().__init__(
            "LogicalMeasurement",
            num_qubits=logical_qubits,
            num_clbits=logical_qubits + 4,
            params=[logical_qubits],
        )
        self.k = logical_qubits

    def _define(self):
        # This is figure 1e in [1]
        k = QuantumRegister(self.k, "k")  # logical qubit register
        t = QuantumRegister(1, "t")
        b = QuantumRegister(1, "b")
        a = QuantumRegister(2, "a")  # ancillas for readout
        cl_k = ClassicalRegister(self.k, "cl_k")
        cl_a = ClassicalRegister(2, "cl_a")
        cl_t = ClassicalRegister(1, "cl_t")
        cl_b = ClassicalRegister(1, "cl_b")
        qc = QuantumCircuit(k, t, b, a, cl_k, cl_t, cl_b, cl_a)

        # Mid circuit reset ancillas
        qc.reset(a)
        qc.h(a[0])

        qc.cx(a[0], t[0])
        qc.cx(a[0], a[1])

        for i in range(self.k):
            qc.cx(a[0], k[i])

        qc.cx(a[0], a[1])
        qc.cx(a[0], b[0])

        qc.barrier()
        qc.h(a[0])
        qc.measure(t, cl_t)
        qc.measure(k, cl_k)
        qc.measure(b, cl_b)
        qc.measure(a, cl_a)

        self._definition = qc
