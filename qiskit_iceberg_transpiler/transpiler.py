r"""A transpiler module that turns a logical quantum circuit into a physical one using the Iceberg code.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister
from qiskit.circuit import Bit as Clbit
from qiskit.circuit import QuantumRegister, Store
from qiskit.circuit.classical import expr, types
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import RXXGate, RZZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import BasisTranslator

from .gates import Initialization, LogicalMeasurement, SyndromeMeasurement, Syndrome


class PhysicalSynthesis(TransformationPass):
    """Pass to convert a circuit of logical operators into physical operators.

    This should be run after inserting syndrome measurements to ensure the measurements
    do not break up logical operations.
    """

    def run(self, dag: DAGCircuit):
        new_dag = dag.copy_empty_like()
        t = self.property_set.get("t", None)[0]
        b = self.property_set.get("b", None)[0]

        # Translate each operator to its logical equivalent. The rules are given in the supplementary text of [1]. We require the circuit to have been transpiled to arbitrary single-qubit pauli rotations and double qubit rotations
        for node in dag.op_nodes():
            match node.name:
                case "x":
                    new_dag.apply_operation_back(node.op, node.qargs)
                    new_dag.apply_operation_back(node.op, t)
                case "rx":
                    new_dag.apply_operation_back(
                        RXXGate(node.params[0]), [t, node.qargs[0]]
                    )
                case "z":
                    new_dag.apply_operation_back(node.op, node.qargs)
                    new_dag.apply_operation_back(node.op, b)
                case "rz":
                    new_dag.apply_operation_back(
                        RZZGate(node.params[0]), [b, node.qargs[0]]
                    )
                case "rxx" | "ryy" | "rzz":
                    # Conveniently, double pauli rotations are the same as their
                    # logical equivalents
                    new_dag.apply_operation_back(node.op, node.qargs)
                case "Initialization" | "LogicalMeasurement" | "SyndromeMeasurement":
                    # Avoid mutating existing iceberg gates that we or the user added
                    new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                case "measure" | "barrier":
                    # Drop measurements because they'll get replaced by logical measurements
                    pass
                case "store":
                    # Keep all stores in the circuit
                    new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                case _:
                    raise Exception(
                        f"Gate `{node.name}` not translatable to a logical operation. Transpile to a basis set of {{rx, rz, rxx, ryy, rzz}}"
                    )

        return new_dag


class InsertSyndromes(TransformationPass):
    """Pass that inserts syndrome placeholders into the circuit

    This is intended to be run after the circuit has been translated to the
    universal basis set, and before synthesizing the logical operators.
    """

    def __init__(
        self,
        add_syndrome_every_n_layers: Optional[int] = None,
        syndrome_checks: Optional[int] = None,
    ):
        """Create the transformation pass

        Args:
            add_syndrome_every_n_layers: Add syndrome measurement every n logical layers (optional)
            syndrome_checks: Add n evenly spaced syndrome checks
        """
        super().__init__()

        self.add_syndrome_every_n_layers = add_syndrome_every_n_layers
        self.syndrome_checks = syndrome_checks

    def run(self, dag: DAGCircuit):
        new_dag = dag.copy_empty_like()
        syndrome_nodes = list(filter(lambda x: x.name == "Syndrome", dag.op_nodes()))

        syndrome_positions = self._validate_arguments(dag)

        # Infer the logical qubits from the first syndrome measurement
        if len(syndrome_nodes) > 0:
            logical_qubits = syndrome_nodes[0].qargs

            for syndrome_op in syndrome_nodes[1:]:
                if logical_qubits != syndrome_op.qargs:
                    raise Exception(
                        "All syndrome measurements must have the same logical qubits"
                    )

        # Assume the entire circuit is logical operations
        else:
            if syndrome_positions is None:
                warnings.warn(
                    "No syndrome measurements found in circuit, and transpiler is not allowed to add syndromes",
                    UserWarning,
                )

            logical_qubits = dag.qubits

        # Save data for next pass
        logical_qubits = list(logical_qubits)
        self.property_set["syndrome_measurements"] = len(syndrome_nodes)
        self.property_set["logical_qubits"] = logical_qubits

        for i, layer in enumerate(dag.layers()):
            if syndrome_positions is not None and i in syndrome_positions:
                new_dag.apply_operation_back(
                    Syndrome(len(logical_qubits)), logical_qubits
                )
                self.property_set["syndrome_measurements"] += 1

            subdag = layer["graph"]
            for node in subdag.op_nodes():
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    def _validate_arguments(self, dag: DAGCircuit):
        """Validates the input arguments

        Returns:
            the logical layer numbers to add syndromes at
        """

        layers = len(list(dag.layers()))

        if (
            self.add_syndrome_every_n_layers is not None
            and self.syndrome_checks is not None
        ):
            raise ValueError(
                "Cannot specify both add_syndrome_every_n_layers and syndrome_checks"
            )
        elif self.add_syndrome_every_n_layers is not None:
            positions = np.arange(0, layers, self.add_syndrome_every_n_layers)
            return positions[1:]  # skip 0
        elif self.syndrome_checks is not None:
            positions = np.linspace(0, layers - 1, self.syndrome_checks + 2)
            return np.round(positions[1:-1]).astype(int)
        else:
            return None


class IcebergSetup(TransformationPass):
    """Pass that adds state prep, logical measurent, and replaces syndrome measurement placeholders with the actual implementation"""

    def __init__(self, use_error_var: bool = True):
        super().__init__()

        self.use_error_var = use_error_var

    def run(self, dag: DAGCircuit):
        new_dag = dag.copy_empty_like()

        # Get the parameters from the previous pass
        logical_qubits = self.property_set["logical_qubits"]
        cl_k = dag.clbits.copy()

        # Add all the iceberg qubits and classical registers
        (t, b, a), (cl_t, cl_b, cl_a) = self._create_iceberg_qubits(new_dag)

        qargs = logical_qubits + t[:] + b[:] + a[:]
        cargs = cl_k + cl_t[:] + cl_b[:] + cl_a[:]

        if self.use_error_var:
            new_dag = self._run_with_var(dag, new_dag, qargs, cargs, cl_a)
        else:
            new_dag = self._run_with_excess_clregs(dag, new_dag, qargs, cargs)

        # Cleanup any unused classical registers
        idle_clbits = list(filter(lambda x: isinstance(x, Clbit), new_dag.idle_wires()))
        new_dag.remove_clbits(*idle_clbits)

        return new_dag

    def _run_with_var(self, dag: DAGCircuit, new_dag: DAGCircuit, qargs, cargs, cl_a):
        k = len(self.property_set["logical_qubits"])

        # Create new variable to hold syndrome error status
        error_flag = expr.Var.new("error", types.Uint(2))
        new_dag.add_declared_var(error_flag)
        self.property_set["error_flag"] = error_flag

        # The instruction we use to copy syndromes to the error variable
        store_instr = Store(error_flag, expr.bit_or(cl_a, error_flag))

        # Logical state preparation
        new_dag.apply_operation_back(Initialization(k), qargs, cl_a)
        new_dag.apply_operation_back(store_instr)

        for node in dag.op_nodes():
            # Replace syndrome placeholders with an actual measurement, followed by a store instruction
            if node.name in {
                "Syndrome",
            }:
                new_dag.apply_operation_back(SyndromeMeasurement(k), qargs, cl_a)
                new_dag.apply_operation_back(store_instr)
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        # Logical measurement
        new_dag.apply_operation_back(LogicalMeasurement(k), qargs, cargs)
        new_dag.apply_operation_back(store_instr)

        return new_dag

    def _run_with_excess_clregs(
        self, dag: DAGCircuit, new_dag: DAGCircuit, qargs, cargs
    ):
        k = len(self.property_set["logical_qubits"])

        synthetic_cregs = []
        syndrome_measurements = self.property_set["syndrome_measurements"] + 2
        for i in range(syndrome_measurements):
            cl = ClassicalRegister(2, f"cl_a{i}")
            new_dag.add_creg(cl)
            synthetic_cregs.append(cl)

        # Logical state preparation
        new_dag.apply_operation_back(Initialization(k), qargs, synthetic_cregs[0])

        # Copy from previous pass, using a new synthetic register for each measurement
        idx = 1
        for node in dag.op_nodes():
            # Replace syndrome placeholders with an actual measurement
            if node.name == "Syndrome":
                new_dag.apply_operation_back(
                    SyndromeMeasurement(k), qargs, synthetic_cregs[idx]
                )
                idx += 1
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        # Logical measurement
        new_dag.apply_operation_back(
            LogicalMeasurement(k), qargs, cargs[:-2] + synthetic_cregs[-1][:]
        )

        return new_dag

    def _create_iceberg_qubits(
        self, dag: DAGCircuit
    ) -> Tuple[Tuple[QuantumRegister, ...], Tuple[ClassicalRegister, ...]]:
        t = QuantumRegister(1, "t")
        dag.add_qreg(t)
        self.property_set["t"] = t

        b = QuantumRegister(1, "b")
        dag.add_qreg(b)
        self.property_set["b"] = b

        a = QuantumRegister(2, "a")
        dag.add_qreg(a)
        self.property_set["a"] = a

        cl_a = ClassicalRegister(2, "cl_a")
        dag.add_creg(cl_a)
        self.property_set["cl_a"] = cl_a

        cl_t = ClassicalRegister(1, "cl_t")
        dag.add_creg(cl_t)
        self.property_set["cl_t"] = cl_t

        cl_b = ClassicalRegister(1, "cl_b")
        dag.add_creg(cl_b)
        self.property_set["cl_b"] = cl_b

        return (t, b, a), (cl_t, cl_b, cl_a)


def get_iceberg_passmanager(
    add_syndrome_every_n_layers: Optional[int] = None,
    syndrome_checks: Optional[int] = None,
    use_error_var: bool = True,
):
    """Passmanager that embeds circuits in the [[k+2, k, 2]] iceberg code

    Args:
        add_syndrome_every_n_layers: Add syndrome measurement every n layers (optional)
        syndrome_checks: Add n evenly spaced syndrome checks (optional)
        use_error_var: Use error variable for syndrome measurement. If True, we bitwise
            OR syndrome measurements to a variable register. If False, we add a new classical register for each syndrome measurement. This can be helpful if a simulator or hardware doesn't support the `Store` instruction.
    """

    basis = ["x", "z", "rx", "rz", "rxx", "ryy", "rzz", "Syndrome"]

    iceberg_pm = PassManager(
        [
            BasisTranslator(
                SessionEquivalenceLibrary,
                target_basis=basis,
            ),
            # Todo: we could perform optimization of the logical program here
            InsertSyndromes(
                add_syndrome_every_n_layers=add_syndrome_every_n_layers,
                syndrome_checks=syndrome_checks,
            ),
            IcebergSetup(use_error_var),
            PhysicalSynthesis(),
        ]
    )
    return iceberg_pm


def transpile(
    inputs,
    add_syndrome_every_n_layers: Optional[int] = None,
    syndrome_checks: Optional[int] = None,
    use_error_var: bool = True,
):
    """Transpiles quantum circuits to the iceberg code

    Args:
    add_syndrome_every_n_layers: Add syndrome measurement every n layers (optional)
    syndrome_checks: Add n evenly spaced syndrome checks (optional)
    use_error_var: Use error variable for syndrome measurement. If True, we bitwise
        OR syndrome measurements to a variable register. If False, we add a new classical register for each syndrome measurement. This can be helpful if a simulator or hardware doesn't support the `Store` instruction.
    """

    pm = get_iceberg_passmanager(
        add_syndrome_every_n_layers=add_syndrome_every_n_layers,
        syndrome_checks=syndrome_checks,
        use_error_var=use_error_var,
    )
    return pm.run(inputs)
