# Quantum Vision Transformer (HViT)

This project implements a **Hybrid Vision Transformer (HViT)** that integrates quantum circuits into transformer-based architectures for enhanced attention mechanisms and classification tasks. The project consists of many main files with two edited being:

1. **`circuits.py`**: Defines quantum circuit architectures and their integration with PyTorch.
2. **`model.py`**: Implements the transformer and HViT model using hybrid attention layers.

---

## **1. circuits.py**

This file focuses on quantum circuit architectures and their use as differentiable modules within neural networks.

### **Key Components**

### **1.1 QLayer**
The `QLayer` class serves as a wrapper to convert quantum circuits into differentiable PyTorch modules. This enables backpropagation through quantum circuits.

- **`__init__`**: Initializes the quantum layer with trainable parameters.
- **`forward`**: Defines the forward pass by applying the quantum circuit to input data.

### **1.2 Quantum Circuits**
Various quantum circuits are implemented for encoding data and computing attention:

- **`loader_bs`**: Prepares quantum states using beam splitters.
- **`mmult_bs`**: Performs matrix multiplication with beam splitter circuits.
- **`rbs`**: Applies Hadamard and controlled rotation gates.
- **`vector_loader` & `matrix_loader`**: Encode vectors and matrices into quantum states.
- **`compute_attention_element`**: Computes individual attention scores using quantum states.
- **`compute_attention`**: Aggregates attention scores across multiple elements.

### **1.3 Attention Measurement Circuits**
Quantum circuits for measuring query, key, and value embeddings in the attention mechanism:

- **`measure_query_key`**: Encodes query and key data, then computes expectations.
- **`measure_value`**: Encodes value data and computes its quantum expectation.

---

## **2. model.py**

This file builds the hybrid and classical transformer architectures, integrating quantum-enhanced attention mechanisms.

### **Key Components**

### **2.1 Encoder Layers**
Three encoder types are implemented:

1. **`EncoderLayer_hybrid1`**: Uses quantum circuits for multi-head attention and the feedforward merger layer.
2. **`EncoderLayer_hybrid2`**: Combines quantum-enhanced attention with a classical feedforward network.
3. **`EncoderLayer`**: Implements classical multi-head attention and feedforward network.

### **2.2 Attention Mechanisms**
The **`AttentionHead_Hybrid2`** class calculates attention using quantum layers:

- **`QLayer`** for computing query, key, and value.
- Scaled softmax for attention computation.

### **2.3 Transformer Architecture**
The **`Transformer`** class integrates positional embeddings, encoder layers, and attention mechanisms. It supports both hybrid and classical attention mechanisms and multiple classification types (e.g., CLS token-based).

### **2.4 Hybrid Vision Transformer (HViT)**
The **`HViT`** class extends the transformer with a fully connected neural network for classification:

- Combines quantum-enhanced attention with classical classification layers.
- Supports configurable embedding dimensions, layer counts, and attention types.

### **2.5 Utilities**
- **`construct_FNN`**: Dynamically builds fully connected layers for classical processing.
- Positional embedding for tokenized data.

---

## **How It Works**

1. **Quantum Circuits**: Quantum circuits encode input data and compute attention scores in the transformer model.
2. **Hybrid Attention**: Combines quantum attention layers with classical layers for scalability and robustness.
3. **Classification**: The HViT model processes tokenized data and outputs predictions via a fully connected classifier.

---

## **Dependencies**
The project relies on:
- PyTorch
- TensorCircuit (for quantum circuits)
- JAX (for backend computation)

---

## **Usage**

1. Define a quantum circuit in `circuits.py`.
2. Integrate the circuit in `model.py` using `QLayer`.
3. Customize the transformer architecture in the `HViT` class.
4. Train the model using PyTorch-compatible datasets.

An example of usage can be found in Final.ipynb.


---

## **Results**

The result of the changes in the two files above led to a speed improvement of 33% with a training and validation accuracy of 25% for 10 epochs as well as a 50% decrease in the wrongly predicted ratio.

---

This architecture bridges the gap between classical and quantum machine learning, offering a versatile framework for hybrid quantum-classical computing.




License
This project is open-source and available for modification and distribution under the MIT License.
