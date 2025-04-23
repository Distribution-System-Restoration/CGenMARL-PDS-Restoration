# Conditional Generative Multi-Agent Soft Actor-Critic for Power Distribution System Restoration

This repository provides the simulation environments for our research on **multi-agent power distribution system restoration** using reinforcement learning. It supports reproducible experiments on IEEE test feeders with switch-level actions and DER constraints.

> ⚠️ **Note:** This repo contains **only the environments** used in the paper. The proposed CGenMARL algorithm will be made available after paper acceptance.

---

## 🔍 Project Overview

We model post-outage restoration as a **multi-agent sequential decision-making** problem. The power distribution system is divided into **microgrids**, each managed by a decentralized agent observing and operating local switches. The goal is to **maximize critical load restoration** using limited DERs while satisfying grid constraints such as voltage stability and capacity limits.

---

## 🧠 Methodology Summary

- **Algorithm:** Soft Actor-Critic (SAC) with Decentralized Actor - Centralized Critic (DACC)
- **Control Space:** Switch-level discrete actions
- **Feedback:** Shared global reward + constraint-aware termination
- **Simulation:** Power flow and constraint checking via OpenDSS

---

## 🔑 Key Features

This repository provides custom simulation environments for reinforcement learning-based restoration studies. Key environment capabilities include:

- **Multi-Agent Architecture Support:** Agents control local microgrids independently with system-wide evaluation.
- **IEEE Test Feeders:** Includes IEEE 123-bus and IEEE 8500-node systems with pre-defined DER locations and load profiles.
- **Discrete Switch Actions:** Agents select from "turn on," "turn off," or "no-action" per switch at each step.
- **Constraint-Aware Evaluation:** Power flow is evaluated using OpenDSS after every joint action to enforce:
  - Power balance
  - Voltage limits
  - Line capacity
- **Feasibility-Based Termination:** Episodes terminate upon constraint violations to emulate real-world grid safety.
- **Critical Load Prioritization:** Buses have assigned importance weights to reflect realistic load restoration objectives.
- **Shared Reward Signal:** Encourages cooperation across agents by linking reward to global restoration success and constraint satisfaction.
- **Pythonic Interface:** Easy-to-use API with `reset`, `step`, and `sample_action` methods compatible with RL pipelines.
- **Fully Extensible:** Modular design allows modification of constraints, feeders, DERs, or reward structure for custom use cases.




## 📦 Prerequisites

To run the environment:

- Python ≥ 3.8
- PyTorch (for downstream RL integration)
- [`opendssdirect.py`](https://github.com/dss-extensions/dss_python)
