# Local D&D 5e RAG Agent (AMD GPU Edition)

A local AI chatbot that answers D&D 5e questions using a RAG (Retrieval Augmented Generation) pipeline and performs dice rolls using custom tools. 

**Tech Stack:** Python, TensorFlow, LangChain, ChromaDB, Ollama.
**Hardware:** Optimized for AMD Radeon RX 6600 (Linux).

---

## ⚡ Quick Start (The "Magic Command")

Since we are not running Ollama as a system service, you must run the server manually to force it to use the AMD GPU (Vulkan Mode).

**1. Start the Brain (Terminal 1)**
Keep this window open while chatting!
```bash
# Forces Ollama to use Vulkan (Video Game Drivers) for the GPU
OLLAMA_VULKAN=1 ollama serve


# Install Python dependencies
pip install -r requirements.txt

# Download the LLM (4.7 GB)
ollama pull llama3.1