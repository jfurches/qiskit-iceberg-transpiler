import pytest
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as Sampler
from scipy import stats
from qiskit import generate_preset_pass_manager
from qiskit.transpiler import StagedPassManager
from qiskit_aer import AerSimulator

from qiskit_iceberg_transpiler import Syndrome, get_iceberg_passmanager
from qiskit_iceberg_transpiler.util import get_good_counts


@pytest.fixture(scope="module")
def ghz_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()
    return qc


def aer_passmanager(**kwargs):
    return StagedPassManager(
        ["iceberg", "transpile"],
        iceberg=get_iceberg_passmanager(**kwargs, use_error_var=False),
        transpile=generate_preset_pass_manager(
            optimization_level=3, backend=AerSimulator()
        ),
    )


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


class TestCircuits:
    def test_noiseless(self, ghz_circuit):
        physical_circuit = aer_passmanager(syndrome_checks=1).run(ghz_circuit)

        sampler = Sampler(default_shots=1024)
        result = sampler.run([physical_circuit]).result()[0]
        counts = get_good_counts(result)

        # Check no samples errored, and that we get the proper ghz state
        assert sum(counts.values()) == 1024
        assert counts["0000"] != 0
        assert counts["1111"] != 0
        assert counts["0000"] + counts["1111"] == 1024

        # Check for statistical deviations from the state
        res = stats.chisquare(list(counts.values()))
        assert res.pvalue > 0.05
