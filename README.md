# tensor-tome
**a local, hybrid-architecture d&d bot**

tensor-tome is an offline ai assistant that helps with tabletop rules and dice rolling. 

it used to be a simple script, but I refactored it to use **langgraph**. this means instead of just a loop, it has a proper state machine (a graph) to decide what to do next.

it's built on a **hybrid architecture**:
1. first, a tiny tensorflow classifier checks your intent (are you saying hi? rolling dice? asking a rule?).
2. then, **langgraph** routes you to the right node:
   - **dice node**: deterministic math. no ai hallucinations here.
   - **chat node**: just friendly vibes.
   - **rag node**: searches your pdfs locally and answers questions using ollama.

## features

- **metadata filtering**: looks for rules in the player's handbook, not random monster lore.
- **hardware optimized**: forces vulkan on linux so your amd gpu actually does the work.
- **privacy**: runs 100% offline. no keys, no cloud, no spying.
- **langgraph**: uses a modern, graph-based flow. easier to extend later.

## tech stack

- python 3.12
- langgraph & langchain
- tensorflow (keras) for intent classification
- ollama (llama 3.2)
- chromadb (vector store)

## setup

### 1. prerequisites
- python 3.10+
- ollama installed locally
- a gpu (optional but nice)

### 2. install
```bash
# clone it
git clone https://github.com/yourusername/tensor-tome.git
cd tensor-tome

# venv setup
python -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# get the model
ollama pull llama3.2
```

## running it

since we aren't using a system service for ollama, you might need to run it manually to get gpu acceleration working right on some systems.

**1. start the brain**
```bash
# force vulkan mode if you have an amd gpu
OLLAMA_VULKAN=1 ollama serve
```

**2. start the bot**
```bash
python chat_llm.py
```

that's it. have fun.
