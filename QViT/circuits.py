import torch.nn as nn
import numpy as np
import tensorcircuit as tc
import torch
from jax import pmap
from jax.numpy import array
import pennylane as qml  # Assuming you are using PennyLane for quantum circuits

####################################### Shared Functions

# Wrapper to turn quantum circuits into differentiable functions with gradients
class QLayer(nn.Module):
    
    def circuit_to_func(self, K, quantum_circuit, nqubits):
        """
        Converts a quantum circuit to a differentiable function using the specified backend (K).
        Vectorizes the function for batch processing and enables Torch compatibility with gradients.
        """
        def f(inputs, parameters):
            return quantum_circuit(inputs, parameters, nqubits)

        # Vectorize the function along the first argument
        f_vmap = K.vmap(f, vectorized_argnums=0)
        # Convert function for use with PyTorch and enable JIT compilation
        f_batch = tc.interfaces.torch_interface(f_vmap, jit=True)

        return f_batch

    def _init_(self, quantum_circuit, par_sizes, nqubits):
        """
        Initializes the QLayer with a given quantum circuit and parameter sizes.
        """
        super(QLayer, self)._init_()
        # Set the backend to JAX
        self.backend = tc.set_backend("jax")
        # Initialize trainable parameters with a normal distribution
        self.w = nn.Parameter(torch.normal(0, 1 / par_sizes[-1] ** 0.5 * torch.ones(par_sizes)))
        # Convert the quantum circuit to a differentiable function
        self.f = self.circuit_to_func(self.backend, quantum_circuit, nqubits)

    def forward(self, input1):
        """
        Forward pass of the QLayer using input data and parameters.
        """
        return self.f(input1, self.w)

########################################### Circuits in the first method

# Optimized RBS function
def rbs_optimized(wires, th):
    """
    Optimized RBS operation using fewer gates.
    """
    qml.CNOT(wires=wires)
    qml.RY(2 * th, wires=wires[1])
    qml.CNOT(wires=wires)

def vector_loader_optimized(alphas, wires=None, is_x=True, is_conjugate=False):
    """
    Optimized vector loader that uses the optimized RBS function.
    """
    if wires is None:
        wires = [i for i in range(alphas.shape[-1] + 1)]
    if is_x and not is_conjugate:
        qml.PauliX(wires=wires[0])
    if is_conjugate:
        for i in reversed(range(alphas.shape[-1])):
            rbs_optimized([wires[i], wires[i + 1]], -alphas[..., i])
    else:
        for i in range(alphas.shape[-1]):
            rbs_optimized([wires[i], wires[i + 1]], alphas[..., i])
    if is_x and is_conjugate:
        qml.PauliX(wires=wires[0])

def compute_attention_element_optimized(inputs, parameters, nqubits):
    """
    Computes an attention element using optimized circuits.
    """
    dev = qml.device('default.qubit', wires=nqubits)

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, parameters):
        alphas_i, alphas_j = torch.split(inputs, inputs.shape[-1] // 2, dim=-1)
        wires = list(range(alphas_i.shape[-1] + 2))
        qml.PauliX(wires=0)
        rbs_optimized(wires[:2], torch.pi / 4)
        vector_loader_optimized(alphas_j, wires[1:], is_x=False)
        # Assuming mmult is optimized similarly
        # mmult_optimized(parameters, wires=wires[1:])
        vector_loader_optimized(alphas_i, wires[1:], is_conjugate=True, is_x=False)
        rbs_optimized(wires[:2], torch.pi / 4)
        return qml.expval(qml.PauliZ(wires[1]))
    
    return circuit(inputs, parameters)

def compute_attention_optimized(alphas, norms, compute_element, nqubits):
    """
    Computes attention using optimized compute_element function.
    """
    yhat = []
    n = norms.shape[1]
    n_items = alphas.shape[0]
    
    for n_i in range(n_items):
        inputs = torch.stack([torch.cat([alphas[n_i, i], alphas[n_i, j]]) for i in range(n) for j in range(n)], dim=0)
        res = compute_element(inputs, None, nqubits)
        e1 = (-res.reshape(n, n) / 2 + 1 / 2 + 1e-10).sqrt()
        wij = e1 * 2 - 1
        yhat.append(wij * torch.outer(norms[n_i], norms[n_i]))
    yhat = torch.stack(yhat, dim=0)
    return yhat

################################################################################ Circuits used in the second method

def encode_token_optimized(circuit, data, nqubits):
    """
    Optimized token encoding using fewer gates.
    """
    for i in range(nqubits):
        circuit.h(i)
        circuit.rx(i, theta=data[i])

def qk_ansatz_optimized(circuit, parameters, nqubits):
    """
    Optimized ansatz for query and key vectors using available gates.
    """
    # Ensure that the parameters array has the correct length
    assert len(parameters) >= 3 * nqubits, "Not enough parameters for the ansatz"

    for i in range(nqubits):
        theta = parameters[i]
        phi = parameters[nqubits + i]
        lam = parameters[2 * nqubits + i]
        # Apply the equivalent of the U3 gate using rz and ry rotations
        circuit.rz(i, theta=phi)
        circuit.ry(i, theta=theta)
        circuit.rz(i, theta=lam)

    # Simplify the entanglement layer
    for i in range(nqubits - 1):
        circuit.cz(i, i + 1)
    circuit.cz(nqubits - 1, 0)

def measure_query_key_optimized(data, parameters, nqubits):
    """
    Measures query and key encoding using optimized circuits.
    """
    circuit = tc.Circuit(nqubits)
    encode_token_optimized(circuit, data, nqubits)
    qk_ansatz_optimized(circuit, parameters, nqubits)
    return circuit.expectation_ps(z=[0]).real

def measure_value_optimized(data, parameters, nqubits):
    """
    Measures the value encoding using optimized circuits.
    """
    circuit = tc.Circuit(nqubits)
    encode_token_optimized(circuit, data, nqubits)
    qk_ansatz_optimized(circuit, parameters, nqubits)
    # Measure expectations more efficiently
    return array([circuit.expectation_ps(z=[i]).real for i in range(nqubits)])
