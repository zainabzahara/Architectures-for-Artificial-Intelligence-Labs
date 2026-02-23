# Architectures for Artificial Intelligence: Hardware-Aware AI Portfolio
**Master’s in Electronics Engineering | University of Bologna**

This repository showcases my implementation of the **APAI 2025** lab sequence, focused on the design and deployment of Deep Neural Networks (DNNs) on **Parallel Ultra Low Power (PULP)** architectures.

---

## 🛠️ Technical Focus & Expertise

* **Edge AI Deployment**: Successfully transitioned models from **PyTorch/ONNX** frameworks to C-based execution on **RISC-V** platforms.
* **Hardware Acceleration**: Utilized **NE16 hardware accelerators** to optimize and speed up convolution operations.
* **Model Optimization**: Applied **Post-Training Quantization (8-bit)** and **Tiling** strategies to fit complex DNNs into constrained L1/L2 memory.
* **On-Device Learning (ODL)**: Implemented **TinyTransformers** and local training modules for autonomous, low-power IoT nodes.

---

## 📂 Laboratory Solutions Breakdown

| Lab Module | Core Technical Implementation |
| :--- | :--- |
| **01-02: Embedded Basics** | Developed optimized C kernels for vector operations; profiled cycle-efficiency using **GVSOC**. |
| **03: DNN Shrinking** | Executed model compression and **8-bit quantization** to minimize memory footprint. |
| **04-06: TinyML & ODL** | Designed **TinyTransformers** and implemented **On-Device Learning** protocols. |
| **07-09: PULP & NE16** | Engineered memory **tiling** logic and integrated **NE16 accelerators** for high-efficiency inference. |
| **10: End-to-End** | Validated the full deployment pipeline, from training to hardware-level execution. |



---

## 🏗️ Environment & Tools

* **Platform**: PULP-SDK / RISC-V.
* **Infrastructure**: Developed within a **Docker** environment using **Dev Containers** for consistent toolchain management.
* **Simulators**: Cycle-accurate profiling conducted via **GVSOC**.

---

### Acknowledgments
*Original templates and course materials provided by the **EEESlab** (Energy-Efficient Embedded Systems Lab) at the **University of Bologna**.*
