# Lab 06: On-Device Learning (ODL) on PULP architectures
**Master’s in Electronics Engineering | University of Bologna**

## 📌 Overview
This lab involved implementing a full **On-Device Learning (ODL)** workload on a PULP-based System-on-Chip (SoC). The project focused on training a Deep Neural Network (DNN) locally on-chip, utilizing the **PULP-TrainLib** to manage backpropagation and gradient descent within strict memory and power constraints.



## 🛠️ Technical Implementation
I developed the C-based training pipeline for a DNN architecture consisting of Conv2D, ReLU, and Fully-Connected layers:

* **Forward Kinematics & Loss**: Implemented the forward prediction pass and coded a custom **MSE (Mean Squared Error)** Loss function to calculate prediction error against PyTorch-generated "Golden Models".
* **Backward Step (Backpropagation)**: Engineered the reversed sequence of operations to propagate gradients from the output back to the input, computing weight gradients for each layer.
* **Linear Layer Optimization**: Completed the primitives for the Fully-Connected layer, implementing matrix-multiplication-based gradients for weights and inputs.
* **Weight Update & Convergence**: Integrated a **Gradient Descent** optimizer to update weights using local blobs, achieving model convergence over 400 epochs.

## 📊 Performance Profiling
Using PULP's internal Performance Counters and `stats.h`, I analyzed the efficiency of the training loop:
* **Latency Analysis**: Measured clock cycles for both forward and backward passes.
* **Throughput**: Benchmarked **MAC/cycle** efficiency to identify bottlenecks in the backpropagation steps.
* **IPC (Instructions Per Cycle)**: Monitored instruction retirement to ensure optimal utilization of the RISC-V cluster.

## 🚀 Environment & Tools
* **Library**: [PULP-TrainLib](https://github.com/pulp-platform/pulp-trainlib).
* **Language**: Embedded C.
* **Simulator**: GVSOC (PULP Virtual Platform).

---
### Acknowledgments
*Original lab templates and course materials provided by **EEESlab** (Energy-Efficient Embedded Systems Lab) at the **University of Bologna**.*
