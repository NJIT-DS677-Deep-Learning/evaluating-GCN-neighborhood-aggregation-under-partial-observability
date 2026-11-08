# Semi-Supervised Classification with Graph Convolutional Networks (GCNs)

## Course
* **Deep Learning:** [DS677]
* **Instructors:**
* Prof. Ioannis Koutis
* Prof. Akshay Rangamani

---

## Project Title
**Evaluating GCN Neighborhood Aggregation Under Partial Observability: From Citation Graphs to Multi-Agent Reinforcement Learning**

---

## Project Claim Statement
> "We claim that GCN neighborhood aggregation (Kipf & Welling, 2017) improves performance over an otherwise identical graph-free model when neighbors hold information a node or agent cannot observe itself. Specifically, we claim it improves test accuracy by at least 10 percentage points over an MLP on the Cora citation dataset (Planetoid split) and achieves significantly higher mean episode reward (Welch's t-test, p < 0.05 over 5 seeds) over an IPPO baseline with MLP policies in a 3-agent VMAS Navigation benchmark environment where the communication radius exceeds the agents' sensing range (R_comm > R_sense). We validate that performance gains derive strictly from the graph structure via an edge-rewiring robustness probe."

---

## Team Members
* **Wenzhe Song** ([ws333@njit.edu](mailto:ws333@njit.edu) / [wzs0028@auburn.edu](mailto:wzs0028@auburn.edu))
* **Ajay Chanderjit** ([aoc3@njit.edu](mailto:aoc3@njit.edu)
* **Rich Wertz** ([rtw7@njit.edu](mailto:rtw7@njit.edu)

---

## Benchmarks & References
* **Paper:** [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, ICLR 2017)](https://arxiv.org/abs/1609.02907)
* **Dataset (Cora):** [PyTorch Geometric Planetoid Split](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html)
* **Environment (MARL):** [VMAS: Vectorized Multi-Agent Simulator](https://github.com/proroklab/VectorizedMultiAgentSimulator)
