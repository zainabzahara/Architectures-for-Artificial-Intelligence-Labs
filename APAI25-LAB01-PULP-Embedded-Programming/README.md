# Lab 01: Embedded Programming & Hardware Profiling on PULP
**Master’s in Electronics Engineering | University of Bologna**

## 📌 Overview
This lab focuses on the implementation and performance analysis of foundational embedded kernels on the **Parallel Ultra Low Power (PULP)** multi-core platform. The objective was to develop low-level C code and utilize cycle-accurate simulation to understand the hardware-software interface.



## 🛠️ Technical Implementation
I successfully implemented and profiled several core kernels, focusing on arithmetic efficiency and memory management:
* **Vector Operations**: Developed optimized C kernels for vector summation.
* **Matrix-Vector Multiplication**: Implemented a matrix-vector product kernel, ensuring proper data alignment for the PULP architecture.
* **Performance Profiling**: Utilized **GVSOC** (PULP Virtual Platform) and hardware counters to measure:
    * **MAC Operations**: Total Multiply-Accumulate operations executed.
    * **Clock Cycles**: Total execution time for each kernel.
    * **CPI Analysis**: Evaluation of Cycles Per Instruction to identify execution bottlenecks.

## 🚀 Environment & Tools
* **Architecture**: PULP (RISC-V multi-core).
* **Toolchain**: PULP SDK and GVSOC simulator.
* **Language**: Embedded C.

---
### Acknowledgments
*Original lab templates and course materials provided by **EEESlab** (Energy-Efficient Embedded Systems Lab) at the **University of Bologna**.*
