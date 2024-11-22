# Quantum Vision Transformer

Overview

Quantum Vision Transformation (QViT) makes the use of quantum circuits for machine learning classification tasks. 
Here, we have applied QViT to the MNIST dataset for classification. Our work has been built up from the work 
done and code created by Eyup Bedirhan Unlu in the QViT_HEP_ML4Sci (https://github.com/EyupBunlu/QViT_HEP_ML4Sci) project 
which is based on the paper, “Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics”
(https://doi.org/10.48550/arXiv.2402.00776).

Quantum Circuits and Functions

qk_ansatz(circuit, parameters, nqubits): 
This function encodes the quantum circuit for key and query vectors by using quantum gate rotations and the CNOT gate.

encode_token(circuit, data, nqubits):
This functions encodes classical data onto the quantum circuit using quantum rotation gates

circuit_to_func(self, K, quantum_circuit, nqubits):
This function converts the quantum circuit to a differentiable function. It vectorizes the function for batch 
processing and enables Torch compatibility with gradients

__init__(self, quantum_circuit, par_sizes, nqubits):
This function initializes the QLayer class with a quantum circuit and parameters sizes.

Other relevant supporting functions

measure_query_key(data, parameters, nqubits):
Measures query and key encoding in the quantum circuit and returns expectation.

measure_value(data, parameters, nqubits):
Measures the value encoding in the quantum circuit and returns expectations for all qubits.


Model Evaluation
The performance of this QViT classification model is evaluated using Cross Entropy Loss functions, 
confusion matrices and ROC curves.


License
This project is open-source and available for modification and distribution under the MIT License.
