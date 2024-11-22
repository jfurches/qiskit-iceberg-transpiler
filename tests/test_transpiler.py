import pytest
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as Sampler
from scipy import stats
from qiskit import generate_preset_pass_manager
from qiskit.transpiler import StagedPassManager
from qiskit_aer import AerSimulator

from qiskit_iceberg_transpiler import Syndrome, get_iceberg_passmanager
from qiskit_iceberg_transpiler.util import get_logical_counts


@pytest.fixture(scope="module")
def ghz_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()
    return qc


@pytest.fixture(scope="module")
def h_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
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
    @pytest.mark.flaky(reruns=5)  # may fail due to statistics
    def test_ghz_noiseless(self, ghz_circuit):
        physical_circuit = aer_passmanager(syndrome_checks=1).run(ghz_circuit)

        sampler = Sampler(default_shots=1024)
        result = sampler.run([physical_circuit]).result()[0]
        counts = get_logical_counts(result)

        # Check no samples errored, and that we get the proper ghz state
        assert counts.shots() == 1024
        assert counts["0000"] != 0
        assert counts["1111"] != 0
        assert len(counts) == 2

        # Check for statistical deviations from the state
        res = stats.chisquare(list(counts.values()))
        assert res.pvalue > 0.05

    @pytest.mark.flaky(reruns=5)  # may fail due to statistics
    def test_h_noiseless(self, h_circuit):
        # This circuit should generate counts either 0000 or 0001. However due to the way the code space is structured, returning just physical bitstrings would yield {0000, 0001, 1110, 1111}. This makes this circuit a better test of the decoding process than the GHZ circuit.
        physical_circuit = aer_passmanager(syndrome_checks=0).run(h_circuit)

        sampler = Sampler(default_shots=1024)
        result = sampler.run([physical_circuit]).result()[0]
        counts = get_logical_counts(result)

        # Check no samples errored, and that we get the proper h state
        assert counts.shots() == 1024
        assert counts["0000"] != 0
        assert counts["0001"] != 0

        # If we didn't decode properly, this would be >2.
        # get_physical_counts returns {
        #   '1 0 1110': 269,
        #   '1 1 1111': 261,
        #   '0 0 0000': 246,
        #   '0 1 0001': 248
        # }
        assert len(counts) == 2

        # Check for statistical deviations from the state. This assumes a uniform distribution
        res = stats.chisquare(list(counts.values()))
        assert res.pvalue > 0.05
