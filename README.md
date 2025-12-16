# DreamPRM-Code 🚀  
**Process Reward Modeling for Code via Chain-of-Functions and Meta Label Correction**

[📄 Paper] | [🌐 Project Page](http://github.com/ruz048/DreamPRM-Code)

---

## 🔍 Overview

**DreamPRM-Code** is a coding-focused **Process Reward Model (PRM)** designed to enhance Large Language Models (LLMs) for program synthesis under **test-time scaling** and **reinforcement learning from feedback**.

Process Reward Models have demonstrated strong effectiveness in mathematical reasoning but remain underexplored for coding tasks due to two key challenges: the absence of natural step decompositions in code and the heavy noise in Monte-Carlo–generated intermediate supervision. DreamPRM-Code addresses both issues by redefining reasoning steps at the level of **functions** and introducing a **meta-learning–based label correction framework**.

When applied to test-time scaling, DreamPRM-Code achieves **state-of-the-art performance on LiveCodeBench**, reaching **80.9 pass@1**, surpassing OpenAI o4-mini and other competitive baselines.

---

## ✨ Key Contributions

- **Chain-of-Functions (CoF) Prompting**  
  Encourages LLMs to generate modular, function-structured programs and defines functions as PRM reasoning steps.

- **Meta-Learning–Based Label Correction**  
  Automatically denoises noisy intermediate PRM labels by leveraging clean final-step unit-test supervision through bi-level optimization.

- **Strong Empirical Performance**  
  Achieves new state-of-the-art results on LiveCodeBench under identical base models.

---

## 🧩 Motivation

Process Reward Models score partial reasoning states and enable:
- Scalable alternatives to human feedback in RLHF
- Test-time scaling via best-of-N selection or tree search

However, unlike mathematical Chain-of-Thought reasoning, code generation lacks a clear notion of “steps.” Naively treating each line as a step leads to excessive computation, while natural-language planning ignores the reasoning embedded in the code itself.

DreamPRM-Code bridges this gap by aligning PRM steps with **software engineering abstractions**, making PRMs practical and effective for coding.

---

## 🔗 Chain-of-Functions as PRM Steps

DreamPRM-Code introduces a **Chain-of-Functions (CoF)** prompting strategy that encourages LLMs to generate code as a sequence of well-defined functions.

Under this paradigm:
- High-level strategy functions are defined first
- Core algorithms and helper utilities are implemented as separate functions
- Each function constitutes a coherent reasoning step for PRM evaluation

This design mirrors human programming practices and allows standard PRM training and inference pipelines—previously used in mathematical reasoning—to be directly applied to code generation.

---

## 🧠 Meta-Learning–Based Label Correction

PRMs are typically trained using Monte-Carlo–sampled labels for intermediate steps, which are often noisy and unreliable. Coding tasks offer a unique opportunity: **final solutions can be automatically evaluated using unit tests**, yielding clean correctness labels.

DreamPRM-Code exploits this property with a **bi-level meta-learning framework**:
- Noisy middle-step labels are treated as learnable variables
- The PRM is trained on these labels in the inner loop
- Performance on clean final-step unit-test data guides label refinement in the outer loop

Through alternating optimization, intermediate labels are progressively denoised, resulting in more robust PRM training and improved downstream performance.

---

## 🧪 Experimental Setup

- **Benchmark**: LiveCodeBench (LeetCode, AtCoder, Codeforces)
- **Training split**: Problems released before 2024-08-01
- **Test split**: Problems released after 2025-02-01 (no overlap)
- **Policy model**: OpenAI o4-mini-high
- **Reward model**: Qwen-2.5-Coder-3B with a classification head
- **Optimization**: Bi-level meta-learning implemented with the Betty library

---

## 📊 Results on LiveCodeBench

| Method | Easy | Medium | Hard | Overall |
|------|------|--------|------|---------|
| o4-mini-high | 100 | 89.7 | 57.4 | 77.1 |
| ORM Scaling | 100 | 89.7 | 62.3 | 79.4 |
| PRM (CoF, no correction) | 100 | 92.3 | 62.3 | 80.2 |
| **DreamPRM-Code** | **100** | **92.3** | **63.9** | **80.9** |

**Key Findings**
- Function-level PRM steps outperform whole-solution reward modeling
- Meta label correction consistently improves PRM effectiveness
- DreamPRM-Code achieves the best overall performance under identical base models

---

## 🏁 Conclusion

DreamPRM-Code demonstrates that **Process Reward Models can be effectively adapted to coding tasks** by:
- Redefining reasoning steps through Chain-of-Functions prompting
- Leveraging unit-test supervision with meta-learning–based label correction

This framework enables robust PRM training and delivers **state-of-the-art test-time scaling performance** on LiveCodeBench.

---

## 📌 Citation

If you find this work useful, please cite:

    @article{dreamprmcode2025,
      title={DreamPRM-Code: Process Reward Modeling for Code via Chain-of-Functions},
      author={Zhang, Ruiyi and others},
      year={2025}
    }

---

## 📬 Contact

For questions or collaborations, please open an issue or visit the project page:  
http://github.com/ruz048/DreamPRM-Code