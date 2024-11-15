r"""A set of modules and classes for working with the [[k+2, k, 2]] iceberg code.
We follow the implementation by Quantinuum [1] to tailor this to trapped-ion processors.

[1] C. Self, M. Benedetti, and D. Amaro, 2024. https://arxiv.org/abs/2211.06703
"""

from collections import Counter
import numpy as np
from qiskit import ClassicalRegister
from qiskit.circuit import Instruction, QuantumCircuit, QuantumRegister, Qubit
from qiskit.primitives import SamplerPubResult


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


def _get_logical_bitstrings(result: SamplerPubResult):
    # Obtain the classical (non-iceberg) registers in order as they appear in the result
    registers = []
    for reg in result.data.keys():
        if reg not in {"cl_t", "cl_b", "cl_a"} and not reg.startswith("cl_a"):
            registers.append(reg)

    # add the iceberg physical qubits at the end
    registers += ["cl_t", "cl_b"]

    total_bitstrings = None
    for reg in registers:
        reg = getattr(result.data, reg)
        min_bits_per_element = np.ceil(np.log2(np.maximum(reg.array.flat, 1)))
        min_bits_per_element = np.maximum(min_bits_per_element, 1)
        max_width = int(max(min_bits_per_element))
        bitstrings = [np.binary_repr(x, width=max_width) for x in reg.array.flat]

        if total_bitstrings is None:
            total_bitstrings = bitstrings
        else:
            total_bitstrings = [
                x + " " + y for x, y in zip(total_bitstrings, bitstrings)
            ]

    return total_bitstrings


def z_stabilizer(result: SamplerPubResult):
    """Calculates the Z stabilizer of the final output, returning ±1

    The iceberg code is defined as being in the joint subspace of Z and X stabilizers. Therefore for any shot with a Z stabilizer of -1, an error occured.

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend

    Returns:
        The Z stabilizer of the circuit as a NumPy array with shape `(shots,)` and entries ±1
    """
    bitstrings = _get_logical_bitstrings(result)
    shots = len(bitstrings)

    parities = np.zeros(shots)
    for i in range(shots):
        parities[i] = bitstrings[i].count("1") % 2

    return 1 - 2 * parities


def has_error(result: SamplerPubResult):
    """Checks each shot for errors by examining syndrome measurements and the sz stabilizer

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend

    Returns:
        A NumPy array with shape `(shots,)` and entries 0 or 1 indicating whether an error occured.
    """
    sz = z_stabilizer(result)

    # Check ancillas. There are 2 possible formats:
    #   1. The circuit could use classical operations (use_error_var = True), so we just
    #      check the error flag != 0.
    #   2. Each syndrome has its own classical register (use_error_var = False), so we
    #      check each register != 0.

    if hasattr(result.data, "error"):
        error = result.data.error.array.flat
    else:
        ancilla_regs = set(filter(lambda x: x.startswith("cl_a"), result.data.keys()))
        error = np.zeros_like(sz, dtype=np.uint8)
        i = 0
        for reg in ancilla_regs:
            reg = getattr(result.data, reg)
            error |= reg.array.flat
            i += 1

    return (sz != +1) | (error != 0)


def get_good_counts(result: SamplerPubResult):
    good_shots = np.logical_not(has_error(result))

    if not np.any(good_shots):
        return {}

    bitstrings = _get_logical_bitstrings(result)

    # Keep good bit strings, trimming off the t and b results
    bitstrings = [bitstrings[i][:-4].strip() for i in np.where(good_shots)[0]]
    return Counter(bitstrings)
