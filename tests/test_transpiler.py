from qiskit_iceberg_transpiler import transpile, get_iceberg_passmanager
from qiskit import QuantumCircuit

import pytest

from qiskit_iceberg_transpiler.code import Syndrome


@pytest.fixture(scope="module")
def ghz_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()
    return qc


class TestTranspiler:
    def test_evenly_spaced_syndromes(self, ghz_circuit):
        pm = get_iceberg_passmanager(syndrome_checks=2)
        physical_circuit = pm.run(ghz_circuit)

        assert pm.property_set["syndrome_measurements"] == 2
        assert physical_circuit.count_ops().get("SyndromeMeasurement", 0) == 2

    def test_fixed_layers(self, ghz_circuit):
        pm = get_iceberg_passmanager(add_syndrome_every_n_layers=16)
        physical_circuit = pm.run(ghz_circuit)

        assert pm.property_set["syndrome_measurements"] == 1
        assert physical_circuit.count_ops().get("SyndromeMeasurement", 0) == 1

    def test_manual_syndrome(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(Syndrome(4), range(4))
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.measure_all()

        pm = get_iceberg_passmanager()
        physical_circuit = pm.run(qc)

        assert pm.property_set["syndrome_measurements"] == 1
        assert physical_circuit.count_ops().get("SyndromeMeasurement", 0) == 1
