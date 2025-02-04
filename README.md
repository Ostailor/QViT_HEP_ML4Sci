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

Scroll down to number 5 to see ways to update the model.

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


Updated

<img width="600" alt="Screenshot 2024-11-24 at 10 05 54 PM" src="https://github.com/user-attachments/assets/ddfe6cec-ccf6-437c-8422-a67d66bb39c4">


---

### **3. Dataset and Preprocessing**

#### **Data Description**
The dataset used in this project is the **MNIST** dataset, which consists of grayscale images of handwritten digits (0-9). Each image has the following characteristics:
- **Source**: Downloaded from PyTorch's `torchvision.datasets.MNIST`.
- **Size**: 
  - Training set contains 60,000 images.
  - Each image has an original resolution of 28x28 pixels.
- **Labels**: Each image is labeled with one of 10 classes corresponding to the digit it represents.

#### **Preprocessing Steps**
1. **Image Transformation**:
   - Resized the original 28x28 images to 14x14 pixels to reduce computational complexity.
   - Converted the images to `torch.float64` for compatibility with the quantum layers.
   - Normalized pixel values to the range `[0,1]` to standardize input.

   Transformations applied:
   ```python
   transforms.Compose([
       transforms.Resize((14,14)),
       transforms.ConvertImageDtype(torch.float64),
       transforms.Normalize(0,1)
   ])


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
- **Accuracy**: Measure of correct predictions. Increased by 25% for 10 epochs.
- **Wrongly predicted ratio**: Decreased by 50% for 10 epochs worth of training.
- **Computation Time**: For the original code it took approximately 15 minutes for 10 epochs. For the updated code it took 10 minutes for 10 epochs.

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
- Integrate Reinforcement Learning into the model.

---
## 6. What has been done so far with RL. 
1. **Policy Gradient-Based RL Module**:
   - A policy gradient algorithm was added to the training pipeline.
   - This module acted as a feedback loop, using rewards to guide hyperparameter adjustments dynamically during training.

2. **Reward Function Design**:
   - A reward function was created based on performance metrics such as accuracy and loss reduction, ensuring meaningful feedback for the RL agent.

3. **Exploration Mechanism**:
   - An exploration mechanism was incorporated to allow the RL module to test a range of parameter values and identify optimal configurations dynamically.

4. **Pipeline Integration**:
   - The RL module was integrated into the existing training pipeline to enable real-time adjustments of hyperparameters during the training process.

## Challenges Encountered

1. **Performance Bottlenecks**:
   - The computational overhead introduced by the RL module, including reward calculations and parameter adjustments, caused frequent timeouts during training.

2. **Excessive Training Duration**:
   - The need for real-time adjustments significantly increased training time, making the approach impractical for larger datasets or models.

3. **Optimization Complexity**:
   - Attempts to streamline the reward function and simplify the RL component were insufficient to resolve the timing and performance issues.

---

## 7. Reference

- [Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics](https://arxiv.org/abs/2402.00776)
