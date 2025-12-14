# Tensor Tome
**A Local, Hybrid-Architecture AI Agent for D&D 5e**

TensorTome is a locally hosted AI assistant designed to answer Dungeons & Dragons 5th Edition rules questions and perform game actions (dice rolling) with high accuracy. 

Unlike standard "wrapper" chatbots, TensorTome utilizes a **Hybrid Architecture**: it uses a lightweight TensorFlow/Keras classifier to route user intents before invoking the heavy LLM. This ensures that deterministic tasks (like math) are handled by code, while semantic queries are handled by a RAG (Retrieval Augmented Generation) pipeline.

---

## Architecture

The system processes user input through a multi-stage pipeline:

1.  **Intent Classification (The Router):** A custom-trained **TensorFlow/Keras** neural network analyzes the input in <0.5s to determine intent (e.g., `rule_lookup`, `roll_dice`, `greeting`).
2.  **Semantic Routing:**
    * **Action Path:** If a dice roll is detected, the system bypasses the LLM and uses a regex-based Python tool for 100% mathematical accuracy.
    * **Knowledge Path:** If a rule question is detected, the system queries **ChromaDB**.
3.  **Strict RAG Retrieval:**
    * Queries are filtered by metadata tags (e.g., searching only the *Player's Handbook* for rules, ignoring the *Monster Manual* to reduce noise).
    * Retrieved chunks are injected into a strict **Llama 3.1** prompt designed to prevent hallucinations (e.g., 3.5e rules).
4.  **Local Inference:** The final response is generated using **Ollama**, optimized for consumer AMD hardware via Vulkan/ROCm.

## Key Features

* **Metadata Filtering:** Solves common RAG noise issues (e.g., searching for "Critical Hits" pulls valid rules instead of random monster stat blocks) by tagging and filtering PDF sources.
* **Hardware Optimization:** Custom configuration to force **AMD Radeon RX 6600** GPU usage on Linux via Vulkan driver overrides, preventing CPU bottlenecks.
* **Agentic Tool Use:** Recognizes natural language commands like *"Roll for initiative"* or *"Damage is 2d6+4"* and executes them programmatically.
* **Privacy First:** Runs 100% offline. No data is sent to OpenAI or cloud providers.

## Tech Stack

* **Core:** Python 3.12, LangChain
* **ML/AI:** TensorFlow (Keras), Ollama (Llama 3.1 8B)
* **Database:** ChromaDB (Vector Store) with HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* **Hardware Acceleration:** AMD ROCm / Vulkan

## Installation

### 1. Prerequisites
* Python 3.10+
* Ollama installed locally
* AMD GPU (Optional, but recommended for speed)

### 2. Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/tensor-tome.git](https://github.com/yourusername/tensor-tome.git)
cd tensor-tome

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Pull the LLM
ollama pull llama3.1
