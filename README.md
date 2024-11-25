# Quantum Vision Transformer (HViT)

## 1. Introduction

### **Motivation**
As the demand for advanced machine learning solutions grows, incorporating quantum computing offers a promising pathway for improving computation and learning efficiency. This project addresses the challenge of enhancing attention mechanisms in transformer models, critical for tasks like image recognition, by leveraging quantum computing.

### **Objectives**
- Develop a hybrid vision transformer (HViT) combining quantum and classical approaches for attention mechanisms.
- Demonstrate the feasibility of quantum-enhanced transformers in real-world datasets.
- Compare hybrid and classical approaches to identify performance advantages.

### **Quantum Advantage**
Quantum machine learning is advantageous due to its ability to handle high-dimensional data encoding, fast parallel computations, and the potential to uncover patterns inaccessible through classical methods. The integration of quantum circuits into transformers enhances attention computations, enabling more efficient and accurate learning.

---

## 2. Methods

### **Framework and Tools**
This project uses:
- **TensorCircuit** for quantum circuit simulations.
- **PyTorch** for model building and training.
- **JAX** as a backend for quantum circuit optimizations.

### **Model Architecture**
The HViT model integrates quantum-enhanced attention heads into a transformer architecture:
1. **Quantum Circuits**: Encode data and compute query-key-value embeddings.
2. **Hybrid Layers**: Combine quantum circuits with classical feedforward networks.
3. **Transformer**: Leverages multi-head attention with positional embeddings.

### **Quantum Circuits**
- **Data Encoding**: Implements beam splitters and rotation gates for encoding.
- **Attention Mechanism**: Uses quantum circuits to compute attention scores based on the inner products of query and key states.
- **Measurement**: Outputs expectation values representing attention weights.

Original
<img width="600" alt="Screenshot 2024-11-24 at 10 01 06 PM" src="https://github.com/user-attachments/assets/5ea55500-f9ed-43ac-910a-83e77b7d3652">


Upgraded
<img width="600" alt="Screenshot 2024-11-24 at 10 05 54 PM" src="https://github.com/user-attachments/assets/ddfe6cec-ccf6-437c-8422-a67d66bb39c4">


---

## 3. Dataset and Preprocessing

### **Data Description**
- **Source**: The dataset used is MNIST
- **Size**: 
- **Features**: 

### **Preprocessing Steps**
- Normalization: Ensures data consistency by scaling values.
- Dimensionality Reduction: Reduces data complexity for quantum circuit compatibility.
- Tokenization: Converts inputs into embeddings for the transformer.

---

## 4. Results

### **Simulations**
- **Simulation**: We used a CPU during training and testing. The changes we made enabled us to increase accuracy in significantly fewer epochs than what was required originally.

### **Key Findings**
- Performance comparisons between old and new circuits as well as vectorization of computing.
- Below is a visualization of accuracy and loss trends during training.

Original
<img width="439" alt="Screenshot 2024-11-24 at 9 41 39 PM" src="https://github.com/user-attachments/assets/fc6f8ed5-792b-4a57-b992-590b1a3645ec">


Updated
<img width="439" alt="Screenshot 2024-11-24 at 9 40 48 PM" src="https://github.com/user-attachments/assets/b15e77cd-915f-42fd-aad3-d6cf0e5b9283">


### **Performance Metrics**
- **Accuracy**: Measure of correct predictions.
- **Wrongly predicted ratio**: 
- **Computation Time**: Old Quantum Circuit versus new Quantum Circuit.

---

## 5. Conclusion

### **Summary**
The HViT model demonstrates the potential of hybrid quantum-classical transformers for improving attention mechanisms, yielding competitive performance on benchmark datasets.

### **Impact**
This work highlights the practical integration of quantum computing into machine learning, paving the way for advancements in data-driven quantum technologies specifically with large datasets.

### **Future Work**
- Explore larger datasets and more efficient quantum circuits.
- Implement the model on actual QPUs for further validation.
- Investigate the scalability of hybrid architectures to larger transformers.

---

## 6. References

- 
