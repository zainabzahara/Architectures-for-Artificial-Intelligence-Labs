# Lab 05: TinyTransformers — Optimizing Attention for Edge AI
**Master’s in Electronics Engineering | University of Bologna**

## 📌 Overview
This lab focused on the implementation and optimization of **Transformer-based architectures** for resource-constrained embedded systems. The objective was to adapt the **Multi-head Self Attention (MHSA)** mechanism and full-stack Transformer blocks to run efficiently on the **PULP** multi-core platform.



## 🛠️ Technical Implementation
I successfully implemented several core Transformer components, focusing on computational efficiency and memory management for low-power hardware:
* **Multi-head Self Attention (MHSA)**: Developed optimized kernels for self-attention, managing the high memory demands of Q, K, and V matrix operations.
* **Full-Stack Transformer Integration**: Orchestrated the integration of Feed-Forward Networks (FFN) and layer normalization within the embedded pipeline.
* **System Initialization**: Managed hardware-specific environment setups and initialization scripts to handle the specialized memory requirements of the Transformer stack.
* **Performance Analysis**: Profiled the execution of attention heads to identify and mitigate computational bottlenecks on the RISC-V cluster.

## 🚀 Environment & Tools
* **Architecture**: PULP (RISC-V multi-core).
* **Implementation**: Embedded C focused on hardware-aware optimization.
* **Simulation**: GVSOC for cycle-accurate performance estimation.

---

### Acknowledgments
*Original lab templates and course materials provided by **EEESlab** (Energy-Efficient Embedded Systems Lab) at the **University of Bologna**.*
