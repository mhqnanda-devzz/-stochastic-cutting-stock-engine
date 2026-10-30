# Stochastic Cutting Stock Engine

> A production-ready optimization engine and rigorous mathematical audit framework for the Stochastic Cutting Stock Problem (SCSP).

## Overview
This repository contains the independent research report and optimization core developed to address structural vulnerabilities in Markov Decision Process (MDP) formulations for the Stochastic Cutting Stock Problem (arXiv:2109.09592). 

It bridges theoretical operations research and practical software engineering by mitigating matrix ill-conditioning, managing non-Markovian demand dynamics, and introducing robust maximal cutting pattern generation.

---

## Key Features
- **Maximal Pattern Generation:** Generates only efficient "extreme" cutting patterns where leftover slack is strictly less than the smallest item length, optimized via backtracking with floating-point tolerance handling.
- **Ridge Regression ($L_2$) Value Approximation:** Stabilizes projected Bellman equations and prevents Runge's phenomenon / matrix singularity in Approximate Dynamic Programming (ADP).
- **Stochastic Demand Simulation:** Models real-world memoryless demand volatility using multiplicative Gaussian shocks.

---

## Project Structure
```text
├── engine.py                                             # Core optimization engine (v0.2 beta)
├── Audit_Report_Cutting_Stock_M._Hexa_Qodrunnanda.pdf    # Full mathematical audit report
├── LICENSE                                               # MIT License
└── README.md                                             # Project documentation

Quick Start (engine.py)
Run the core engine locally (compatible with Termux or standard Python environments):
import numpy as np
from engine import StochasticCuttingStockEngine

# Initialize engine with 100cm stock length and L2 regularization
engine = StochasticCuttingStockEngine(stock_length=100.0, lambda_reg=0.005)

# Generate maximal cutting patterns for specific item demands
item_demands = [25, 15, 10]
maximal_patterns = engine.generate_maximal_cutting_patterns(item_demands)

print(f"Total maximal patterns found: {len(maximal_patterns)}")
print(maximal_patterns)

Author
Muhammad Hexa Qodrunnanda
Principal Operations Research Scientist & Mathematical Auditor
License
Distributed under the MIT License. See LICENSE for more information.

