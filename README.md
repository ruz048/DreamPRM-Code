# DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding 

[📄 Paper] 

---

## 🔍 Overview

DreamPRM-Code is a coding-focused Process Reward Model that enables reliable test-time scaling for LLM coding. It resolves the two main blockers for coding PRMs: (1) missing PRM step definitions and (2) noisy intermediate PRM training labels. For 1, it leverages a chain-of-functions prompt to define PRM steps at the **function** level. For 2, it denoises Monte-Carlo sampled PRM training label with meta-learning guided by unit-test outcomes. 

### Key Mechanics

#### 1. Chain-of-Functions prompting
Inspired by Chain-of-Thought, DreamPRM-Code uses Chain-of-Function (CoF) prompt to steer the LLM toward producing independent code blocks whose logic can be isolated and encapsulated into separate functions.

<strong>Example of generated code under CoF prompting </strong>

```python
(Step-1)
def main():
    '''
    Strategy: Use Dijkstra's algorithm to find the shortest path...
    '''
    # implementation

(Step-2)
def dijkstra(graph, start, end):
    '''
    Implements Dijkstra's algorithm with a min-heap priority queue...
    '''
    # implementation

(Step-3)
def build_graph(n, m):
    '''
    Build adjacency list from stdin input...
    '''
    # implementation
```


#### 2. Meta-learning label correction
Noisy MC-sampled PRM training labels are treated as learnable variables and refined via a meta-learning scheme that is anchored by clean final step rewards, producing more faithful intermediate supervision.

<div align="center">
<figure>
  <img src="figs/flowchart.png" alt="Meta-learning label correction flowchart" width="30%">
</figure>
</div>

#### 3. Experimental setup
We use LiveCodeBench (post-2025-02) as the test set, OpenAI o4-mini-high as the base LLM model, and Qwen-2.5-Coder-3B as the PRM.

#### 📊 Benchmark performance

<div align="center">

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Easy</th>
      <th>Medium</th>
      <th>Hard</th>
      <th>Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gemini-2.5</td>
      <td><strong>100</strong></td>
      <td>82.1</td>
      <td>52.5</td>
      <td>72.5</td>
    </tr>
    <tr>
      <td>O3</td>
      <td><strong>100</strong></td>
      <td>71.8</td>
      <td>57.4</td>
      <td>71.8</td>
    </tr>
    <tr>
      <td>DeepSeek-R1</td>
      <td>99.7</td>
      <td>77.7</td>
      <td>47.2</td>
      <td>68.7</td>
    </tr>
    <tr>
      <td>O4-mini-high</td>
      <td><strong>100</strong></td>
      <td>89.7</td>
      <td>57.4</td>
      <td>77.1</td>
    </tr>
    <tr>
      <td>ORM (o4-mini-high)</td>
      <td><strong>100</strong></td>
      <td>89.7</td>
      <td>62.3</td>
      <td>79.4</td>
    </tr>
    <tr>
      <td>PRM-CoF (o4-mini-high)</td>
      <td><strong>100</strong></td>
      <td><strong>92.3</strong></td>
      <td>62.3</td>
      <td>80.2</td>
    </tr>
    <tr>
      <td><strong>DreamPRM-Code</strong></td>
      <td><strong>100</strong></td>
      <td><strong>92.3</strong></td>
      <td><strong>63.9</strong></td>
      <td><strong>80.9</strong></td>
    </tr>
  </tbody>
</table>

</div>

---

## 🚀 Quick Start

This section provides a minimal end-to-end guide for training and evaluating **DreamPRM-Code** on LiveCodeBench.

---

### 1️⃣ Clone the Repository and Set Up Environment

First, clone the repository and create the conda environment using the provided `environment.yml` file:

    git clone https://github.com/ruz048/DreamPRM-Code.git
    cd DreamPRM-Code
    conda env create -f environment.yml
    conda activate dreamprm-code

---

### 2️⃣ Generate Chain-of-Functions (CoF) Training Data

DreamPRM-Code relies on **Chain-of-Functions (CoF)**–structured code as PRM training data.

Use the following script to generate CoF-style code solutions from the base LLM:

    bash gen_cof.sh

This step produces **function-structured code** that define PRM reasoning steps.  
At this stage, the generated data **does not contain reward labels**.

---

### 3️⃣ Monte-Carlo Label Generation

To obtain initial supervision for PRM training, we perform **Monte-Carlo (MC) sampling** to assign noisy correctness labels to intermediate CoF steps:

    bash gen_cof_label.sh

These labels serve as the starting point for training and will later be **automatically refined** by the meta-learning–based label correction framework.

---

### 4️⃣ Generating Multiple LLM Coding Solution

To generate multiple LLM solutions for trained PRM to select from:

    bash gen_sol.sh

It currently uses OpenAI o4-mini-high to generate solutions, which is the same as original LiveCodeBench  settings. 

### 5️⃣ Meta-Learning–Based Training and Evaluation

With CoF data and MC-sampled labels prepared, you can start training DreamPRM-Code under the bi-level optimization framework:

    bash run_train_eval.sh

This script:
- Trains the PRM on function-level steps
- Performs meta-learning–based label correction
- Automatically evaluates test-time scaling performance after training if multiple LLM solutions have been generated in previous step

---

### 6️⃣ Evaluation with a Trained Checkpoint

If you already have a trained DreamPRM-Code checkpoint, you can directly run evaluation without retraining:

    bash run_eval.sh

This evaluates the PRM under test-time scaling settings on the specified benchmark.

### Pretrained checkpoint

We provide our trained checkpoint of DreamPRM-Code here: [[DreamPRM-Code-ckpt]](https://drive.google.com/file/d/1mt4BzB9laiO7yDHVip0TFSVWOEeUBRB1/view?usp=sharing). Using this checkpoint together with LLM generated solutions, you can directly reproduce our results following instructions in step 6️⃣.

---

## Acknowledgement <a name="acknowledgement"></a>

+ [LiveCodeBench](https://livecodebench.github.io)
+ [DreamPRM](https://github.com/coder-qicao/DreamPRM)
+ [Betty](https://github.com/leopard-ai/betty)


## License <a name="license"></a>
This repository is under [Apache License 2.0](LICENSE.md).


## 📌 Citation

If you find this work useful, please cite:

    @article{zhang2025dreamprmcode,
    title   = {DreamPRM-Code: Function-as-Step Process Reward Model for LLM Coding},
    author  = {Zhang, Ruiyi and Qin, Peijia and Cao, Qi and Xie, Pengtao},
    journal = {arXiv preprint},
    year    = {2025}
    }

---

## 📬 Contact

For questions or collaborations, please contact ruz048@ucsd.edu 
