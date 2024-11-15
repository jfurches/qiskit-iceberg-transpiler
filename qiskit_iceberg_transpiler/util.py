from collections import Counter
import numpy as np
from qiskit.primitives import SamplerPubResult


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
    """Extracts the error-free bit strings from a result and returns the histogram

    Args:
        result: A set of shots from the `SamplerV2` primitive after executing on the backend

    Returns:
        A histogram of error-free samples. If all samples have errors, this will be an empty dict
    """
    good_shots = np.logical_not(has_error(result))

    if not np.any(good_shots):
        return {}

    bitstrings = _get_logical_bitstrings(result)

    # Keep good bit strings, trimming off the t and b results
    bitstrings = [bitstrings[i][:-4].strip() for i in np.where(good_shots)[0]]
    return Counter(bitstrings)
