# Lab 07: Memory Tiling for Edge AI Inference (Part 1)
**Master’s in Electronics Engineering | University of Bologna**

## 📌 Overview
This lab focused on the implementation of **Memory Tiling** strategies for Deep Neural Networks on the **PULP** architecture. The objective was to manage the strict memory hierarchy of the SoC by orchestrating data movement between large, slow L2 memory and small, fast L1 (Tightly Coupled) memory.



## 🛠️ Technical Implementation
I implemented a manual tiling approach to enable the execution of neural kernels that exceed the physical capacity of L1 memory:

* **Tiling Logic Design**: Engineered algorithms to partition large input/output tensors and weight matrices into optimally sized "tiles" that fit within the 64kB L1 memory.
* **Double Buffering & DMA**: (If applicable) Prepared the framework for asynchronous data transfers to overlap computation with communication, maximizing hardware utilization.
* **Pointer Arithmetic**: Developed robust C code to manage offsets and strides when navigating through tiled data structures.
* **Constraint Solving**: Calculated the maximum possible tile dimensions based on the memory footprint of specific DNN layers (Conv2D/Linear).

## 📊 Performance Analysis
* **Memory Efficiency**: Successfully executed workloads that were previously impossible to run in a single-pass L1 allocation.
* **Cycle Profiling**: Evaluated the impact of tiling overhead on total execution cycles using **GVSOC** performance counters.

## 🚀 Environment & Tools
* **Platform**: PULP / RISC-V Multi-core.
* **Tools**: PULP-SDK and GVSOC simulator.
* **Language**: Embedded C.

---
### Acknowledgments
*Original lab templates and course materials provided by **EEESlab** (Energy-Efficient Embedded Systems Lab) at the **University of Bologna**.*
