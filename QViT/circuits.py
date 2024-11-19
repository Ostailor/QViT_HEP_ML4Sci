import torch.nn as nn
import numpy as np
import tensorcircuit as tc
import torch
from jax import pmap
from jax.numpy import array

####################################### Shared Functions

# Wrapper to turn quantum circuits into differentiable functions with gradients
class QLayer(nn.Module):
    
    def circuit_to_func(self, K, quantum_circuit, nqubits):
        """
        Converts a quantum circuit to a differentiable function using the specified backend (K).
        Vectorizes the function for batch processing and enables Torch compatibility with gradients.
        
        Args:
            K: The quantum backend.
            quantum_circuit: Quantum circuit function.
            nqubits: Number of qubits used in the circuit.
        
        Returns:
            f_batch: A batch-processable, differentiable quantum circuit function.
        """
        def f(inputs, parameters):
            return quantum_circuit(inputs, parameters, nqubits)

        # Vectorize the function along the first argument
        f_vmap = K.vmap(f, vectorized_argnums=0)
        # Convert function for use with PyTorch and enable JIT compilation
        f_batch = tc.interfaces.torch_interface(f_vmap, jit=True)

        return f_batch

    def __init__(self, quantum_circuit, par_sizes, nqubits):
        """
        Initializes the QLayer with a given quantum circuit and parameter sizes.

        Args:
            quantum_circuit: The quantum circuit function.
            par_sizes: Size of parameters for the quantum circuit.
            nqubits: Number of qubits used in the circuit.
        """
        super(QLayer, self).__init__()
        # Set the backend to JAX
        self.backend = tc.set_backend("jax")
        # Initialize trainable parameters with a normal distribution
        self.w = nn.Parameter(torch.normal(0, 1 / par_sizes[-1] ** 0.5 * torch.ones(par_sizes)))
        # Convert the quantum circuit to a differentiable function
        self.f = self.circuit_to_func(self.backend, quantum_circuit, nqubits)

    def forward(self, input1):
        """
        Forward pass of the QLayer using input data and parameters.
        
        Args:
            input1: Input data for the quantum layer.

        Returns:
            The output of the quantum function with gradients.
        """
        return self.f(input1, self.w)


########################################### Circuits in the first method
# Circuit Architectures from QViT Paper, included for reference

def loader_bs(X):
    """
    Loads data X into quantum states via beam splitters.
    Applies a PauliX gate to the first wire and normalizes input X.

    Args:
        X: Input data to be loaded as quantum states.
    """
    qml.PauliX(wires=0)
    for i, x in enumerate(X):
        qml.Beamsplitter(X[i] / X.max(), 0, [i, i + 1])

def mmult_bs(parameters, X, length=3):
    """
    Implements matrix multiplication with a quantum beam splitter circuit.

    Args:
        parameters: Circuit parameters.
        X: Input data.
        length: Number of layers for beam splitters.

    Returns:
        Expectation value of PauliZ measurement.
    """
    k = 0
    loader_bs(X)
    for i in range(2 * length - 2):
        j = length - abs(length - 1 - i)
        
        if i % 2:
            for _ in range(j):
                if _ % 2 == 0:
                    qml.Beamsplitter(parameters[k], 0, [_, _ + 1])
                    k += 1
        else:
            for _ in range(j):
                if _ % 2:
                    qml.Beamsplitter(parameters[k], 0, [_, _ + 1])
                    k += 1
    return qml.expval(qml.PauliZ([1]))


# Circuit architectures utilized in the project

def rbs(wires, th):
    """
    Performs an RBS (random basis state) operation with Hadamard, CZ, and RY rotations.

    Args:
        wires: Wires on which to perform the RBS operation.
        th: Rotation angle theta.
    """
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])
    qml.CZ(wires)
    qml.RY(th, wires[0])
    qml.RY(-th, wires[1])
    qml.CZ(wires)
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])

def vector_loader(alphas, wires=None, is_x=True, is_conjugate=False):
    """
    Loads a vector of data into quantum states using the RBS operation.

    Args:
        alphas: The vector data to load.
        wires: Wires on which to apply the operation.
        is_x: Apply PauliX gate if True.
        is_conjugate: If True, apply conjugate operation.
    """
    if wires is None:
        wires = [i for i in range(alphas.shape[-1] + 1)]
    if is_x and not is_conjugate:
        qml.PauliX(wires=wires[0])
    if is_conjugate:
        for i in reversed(range(alphas.shape[-1])):
            rbs([wires[i], wires[i + 1]], -alphas[..., i])
    else:
        for i in range(alphas.shape[-1]):
            rbs([wires[i], wires[i + 1]], alphas[..., i])
    if is_x and is_conjugate:
        qml.PauliX(wires=wires[0])

def matrix_loader(mag_alphas, alphas, mag_wires, wires, is_conjugate=False):
    """
    Loads a matrix into quantum states using magnitude and vector data.

    Args:
        mag_alphas: Norms of matrix rows.
        alphas: Row values of the matrix.
        mag_wires: Wires to store row norms.
        wires: Wires to store row data.
        is_conjugate: If True, applies conjugate operation.
    """
    if not is_conjugate:
        vector_loader(mag_alphas, mag_wires)
        for i in range(len(mag_wires)):
            qml.CNOT([mag_wires[i], wires[0]])
            vector_loader(alphas[i], wires, is_x=False)
            if i != len(mag_alphas):
                vector_loader(alphas[i + 1], wires, is_x=False, is_conjugate=True)
    else:
        for i in reversed(range(len(mag_wires))):
            if i != len(mag_alphas):
                vector_loader(alphas[i + 1], wires, is_x=False, is_conjugate=False)
            vector_loader(alphas[i], wires, is_x=False, is_conjugate=True)
            qml.CNOT([mag_wires[i], wires[0]])
        vector_loader(mag_alphas, mag_wires, is_conjugate=True)

def compute_attention_element(inputs, parameters):
    """
    Computes an attention element by encoding vectors and applying gates.

    Args:
        inputs: Input tensor split into two attention vectors.
        parameters: Parameters for gates.

    Returns:
        Expectation value as the attention score.
    """
    alphas_i, alphas_j = torch.split(inputs, inputs.shape[-1] // 2, dim=-1)
    wires = list(range(alphas_i.shape[-1] + 2))
    qml.PauliX(wires[0])
    rbs(wires[:2], torch.pi / 4)
    vector_loader(alphas_j, wires[1:], is_x=False)
    mmult(parameters, wires=wires[1:])
    vector_loader(alphas_i, wires[1:], is_conjugate=True, is_x=False)
    rbs(wires[:2], torch.pi / 4)
    return qml.expval(qml.PauliZ([wires[1]]))

def compute_attention(alphas, norms, compute_element):
    """
    Computes attention by iterating over elements and normalizing results.

    Args:
        alphas: Attention weight vectors.
        norms: Norm values for normalization.
        compute_element: Function to compute individual attention elements.

    Returns:
        Tensor containing attention scores.
    """
    yhat = []
    n = norms.shape[1]
    n_items = alphas.shape[0]
    
    for n_i in range(n_items):
        res = compute_element(torch.stack([alphas[n_i, [i, j]].flatten() for j in range(n) for i in range(n)], dim=0))
        e1 = (-res.reshape(n, n) / 2 + 1 / 2 + 1e-10).sqrt()
        wij = e1 * 2 - 1
        yhat.append(wij * torch.outer(norms[n_i], norms[n_i]))
    yhat = torch.stack(yhat, dim=0)
    return yhat


################################################################################ Circuits used in the second method

def encode_token(circuit, data, nqubits):
    for i in range(nqubits):
        circuit.rx(i, theta=data[i])

def qk_ansatz(circuit, data, parameters, nqubits):
    for i in range(nqubits):
        circuit.ry(i, theta=parameters[i])
    for i in range(nqubits - 1):
        circuit.cnot(i, i + 1)

def measure_query_key(data, parameters, nqubits):
    circuit = tc.Circuit(nqubits)
    encode_token(circuit, data, nqubits)
    qk_ansatz(circuit, data, parameters, nqubits)
    return circuit.expectation_ps(z=[0]).real

def measure_value(data, parameters, nqubits):
    circuit = tc.Circuit(nqubits)
    encode_token(circuit, data, nqubits)
    v_ansatz(circuit, data, parameters, nqubits)
    return array([circuit.expectation_ps(z=[i]).real for i in range(nqubits)])

