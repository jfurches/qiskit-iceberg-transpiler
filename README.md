# Qiskit Iceberg Transpiler

This is an experimental module for encoding logical circuits defined in Qiskit with the Iceberg quantum error detection code. It largely follows the implementation details from the Quantinuum demonstration [[1](https://arxiv.org/abs/2211.06703)].

The Iceberg code is a $[[k+2, k, 2]]$ detection code, meaning it encodes $k$ logical qubits using $k+2$ physical qubits (and technically 2 more ancillas) [[2](https://errorcorrectionzoo.org/c/iceberg)]. Its logical operations require long-range gates, which makes it more suited for trapped-ion devices with all-to-all coupling maps.

[1] C. Self, M. Benedetti, and D. Amaro, 2024. [https://arxiv.org/abs/2211.06703](https://arxiv.org/abs/2211.06703)

[2] Error correction zoo. [https://errorcorrectionzoo.org/c/iceberg](https://errorcorrectionzoo.org/c/iceberg)


## Installation

The package can be installed from pip with

```
pip install qiskit-iceberg-transpiler
```

As it uses some newer features of Qiskit like classical variables, it requires Qiskit 1.2.x or higher.

## Usage

To get started, you'll need a `QuantumCircuit`, which can be manually constructed, loaded from QASM, or whatever. Everything in this `QuantumCircuit` will be treated as _logical_ operations.

To produce the physical circuit, we can simply use the function `transpile`. We also need to specify where to put syndrome checks. This is demonstrated below,

```python
# Import with a different name to not mix up with qiskit
from qiskit_iceberg_transpiler import transpile as ib_transpile

logical_circuit = QuantumCircuit(...)

# We can put syndrome checks after every `n` logical layers
add_syndrome_every_n_layers = 16

# Or we can choose to evenly space `n` checks. Note that only
# one of these flags should be set.
syndrome_checks = 3

# Transpile to get our circuit made of physical operators, including
# logical state preparation and measurement
physical_circuit = ib_transpile(
    logical_circuit,
    add_syndrome_every_n_layers=add_syndrome_every_n_layers,
    syndrome_checks=syndrome_checks,
)
```

If these default options don't fit your application and you want finer control over the syndrome placement, you can manually add the `Syndrome` placeholder to your circuit, which the transpiler will later replace with real checks.

```python
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_iceberg_transpiler import Syndrome

# Define our logical circuit
k = QuantumRegister(4, 'k')
cl_k = ClassicalRegister(4, 'k')
qc = QuantumCircuit(k, cl_k)

# Start of GHZ circuit
qc.h(k[0])
qc.cx(k[0], k[1])

# Manual syndrome check. Note that all syndrome checks must operate on the same set of qubits (in the same order)
qc.append(Syndrome(4), k)

# Rest of circuit
qc.cx(k[1], k[2])
qc.cx(k[2], k[3])
qc.measure(k, cl_k)
```

See more in the [examples](./examples/getting-started.ipynb).

## Todo

[ ] Make it work as a qiskit transpiler plugin

[ ] Add more tests

[ ] Simulate with noisy models

[ ] Replicate QV tests from [1]
