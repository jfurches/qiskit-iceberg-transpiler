from collections import Counter
from typing import List

import numpy as np
from qiskit.primitives import DataBin, SamplerPubResult
from qiskit.result import Counts


def get_physical_memory(result: SamplerPubResult, kind: str = "good"):
    """Returns the bitstrings across the physical qubits [k] ∪ {t, b}

    Unlike `get_logical_memory`, this returns the values for qubits {t, b}, and
    it does not perform decoding from the code space.

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend
        kind (optional): Filter the bitstrings by error status. See `get_logical_memory`

    Returns:
        memory: Memory of the physical qubits :math:`[k] ∪ {t, b}`, optionally filtered
    """
    # Obtain the classical (non-iceberg) registers in order as they appear in the result, putting t and b at the end.
    registers = _get_logical_regs(result) + ["cl_t", "cl_b"]
    data = result.join_data(registers)

    mask = None
    if kind == "good":
        mask = np.where(~has_error(result))[0]
        data = data.slice_shots(mask)
    elif kind == "bad":
        mask = np.where(has_error(result))[0]
        data = data.slice_shots(mask)

    return data.get_bitstrings()


def get_physical_counts(result: SamplerPubResult, kind: str = "good"):
    """Returns the bitstrings across the physical qubits [k] ∪ {t, b}

    Unlike `get_logical_memory`, this returns the values for qubits {t, b}, and
    it does not perform decoding from the code space.

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend
        kind (optional): Filter the bitstrings by error status. See `get_logical_memory`

    Returns:
        counts: Histogram of the physical qubits :math:`[k] ∪ {t, b}`, optionally filtered
    """

    creg_sizes = [
        (name, reg.num_bits)
        for name, reg in result.data.items()
        if _is_physical_reg(name)
    ]
    memory = get_physical_memory(result, kind=kind)
    memory_slots = sum(size for _, size in creg_sizes)
    return Counts(Counter(memory), creg_sizes=creg_sizes, memory_slots=memory_slots)


def _is_physical_reg(name: str):
    return _is_logical_reg(name) or name in {"cl_t", "cl_b"}


def _is_logical_reg(name: str):
    return name not in {"cl_t", "cl_b", "cl_a"} and not name.startswith("cl_a")


def _get_logical_regs(result: SamplerPubResult):
    return list(filter(_is_logical_reg, result.data))


def get_logical_memory(result: SamplerPubResult, kind: str = "good") -> List[str]:
    """Returns the logical bitstrings on [k] after decoding from the code space

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend
        kind (optional): Filter the bitstrings by error status
            - all: Return all bitstrings
            - good: Return only error-free bitstrings
            - bad: Return only errored bitstrings

    Returns:
        memory: Memory of the logical qubits :math:`[k] = {1, ..., k}`, optionally filtered
    """

    if kind not in {"all", "good", "bad"}:
        raise ValueError("filter must be one of {'all', 'good', 'bad'}")

    # Decode from code space by flipping logical qubits wherever b = 1
    logical_regs = _get_logical_regs(result)
    cl_b = result.data.cl_b.array.flat
    for name in logical_regs:
        reg = getattr(result.data, name)
        reg.array.flat[:] = np.where(
            cl_b, np.bitwise_not(reg.array.flat), reg.array.flat
        )

    data = result.join_data(logical_regs)

    mask = None
    if kind == "good":
        mask = np.where(~has_error(result))[0]
        data = data.slice_shots(mask)
    elif kind == "bad":
        mask = np.where(has_error(result))[0]
        data = data.slice_shots(mask)

    return data.get_bitstrings()


def get_logical_counts(result: SamplerPubResult, kind: str = "good"):
    """Returns the logical histogram on [k] after decoding from the code space

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend
        kind (optional): Filter the bitstrings by error status. See `get_logical_memory`

    Returns:
        logical_counts: Histogram of the logical qubits :math:`[k] = {1, ..., k}`, optionally filtered
    """

    creg_sizes = [
        (name, reg.num_bits)
        for name, reg in result.data.items()
        if _is_logical_reg(name)
    ]

    memory = get_logical_memory(result, kind=kind)
    memory_slots = sum(size for _, size in creg_sizes)
    return Counts(Counter(memory), creg_sizes=creg_sizes, memory_slots=memory_slots)


def z_stabilizer(result: SamplerPubResult = None):
    """Calculates the Z stabilizer of the final output, returning ±1

    The iceberg code is defined as being in the joint subspace of Z and X stabilizers. Therefore for any shot with a Z stabilizer of -1, an error occured.

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend

    Returns:
        The Z stabilizer of the circuit as a NumPy array with shape `(shots,)` and entries ±1
    """
    registers = _get_logical_regs(result) + ["cl_t", "cl_b"]
    databin = result.join_data(registers)
    parities = databin.bitcount() % 2
    return 1 - 2 * parities.astype(int)


def has_error(result: SamplerPubResult):
    """Checks each shot for errors by examining syndrome measurements and the sz stabilizer

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend

    Returns:
        A NumPy array with shape `(shots,)` and entries 0 or 1 indicating whether an error occured.
    """
    sz = z_stabilizer(result=result)

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
