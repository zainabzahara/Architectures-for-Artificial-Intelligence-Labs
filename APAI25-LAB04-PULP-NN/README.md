# Lab 04: PULP-NN — Optimized Neural Network Kernels
**Master’s in Electronics Engineering | University of Bologna**

## 📌 Overview
This lab focused on the implementation and performance evaluation of **PULP-NN**, a library of specialized kernels designed for Deep Learning inference on the **Parallel Ultra Low Power (PULP)** architecture. The objective was to utilize highly optimized assembly-level primitives to accelerate neural network operations on RISC-V multi-core clusters.

[Image of PULP-NN library architecture and optimization layers]

## 🛠️ Technical Implementation
I worked with low-level software stacks to maximize the throughput of neural kernels:
* **Kernel Integration**: Successfully integrated PULP-NN kernels into the application code to replace standard, non-optimized C implementations.
* **SIMD & Parallelization**: Utilized the multi-core cluster and SIMD (Single Instruction, Multiple Data) instructions to parallelize convolution and linear layers.
* **Memory Management**: Optimized the use of L1 (Tightly Coupled L1 memory) to minimize data movement overhead during inference.
* **Efficiency Benchmarking**: Compared execution cycles and energy consumption between standard C code and optimized PULP-NN kernels.

## 🚀 Environment & Tools
* **Library**: PULP-NN (optimized for RISC-V/PULP).
* **Architecture**: Multi-core PULP cluster.
* **Simulation**: GVSOC for cycle-accurate performance estimation.

---
### Acknowledgments
*Original lab templates and course materials provided by **EEESlab** (Energy-Efficient Embedded Systems Lab) at the **University of Bologna**.*
