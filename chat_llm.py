import json
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dice_tool import roll_dice


# LOAD TF & OLLAMA

print("Loading Ollama...")

# 1. Load model, tokenizer, and labels
model = load_model('chat_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc_file:
    lbl_encoder = pickle.load(enc_file)

# 2. Setup Ollama (The LLM)
llm = Ollama(model="llama3.1")

max_len = 20

# load the dnd knowledge
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="dnd_db", embedding_function=embedding_model)


# promt logic

def query_dnd_manuals(question, category_filter=None):
    """Searches the PDF for the answer"""
    print(f"(Searching the rulebooks... Filter: {category_filter})")

    search_kwargs = {"k": 5}

    if category_filter:
        search_kwargs["filter"] = {"category": category_filter}

    
    # fetch top 3 matches
    results = db.similarity_search(question, **search_kwargs)

    valid_sources = list(set([doc.metadata['source_file'] for doc in results]))

    # combine the text from those results
    context_text = "\n\n".join([f"[Source: {doc.metadata['source_file']}] {doc.page_content}" for doc in results])

    # create the RAG prompt
    rag_prompt = f"""
    You are a D&D Assistant. Answer the question using ONLY the context below.
    
    MANDATORY CITATION RULE:
    Every fact you state must be supported by the provided context.
    At the end of your answer, list the Source Files you used.
    
    VALID SOURCES AVAILABLE TO YOU:
    {valid_sources}
    
    CONTEXT:
    {context_text}
    
    QUESTION:
    {question}
    """

    return rag_prompt

# chat loop
print("Hybrid Bot is ready! (Powered by Llama 3.1)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    # TF classification
    seq = tokenizer.texts_to_sequences([user_input]) 
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)
    result = model.predict(padded, verbose=0)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]
    confidence = result[0][np.argmax(result)]
    # --- ADD THIS DEBUG LINE ---
    print(f"DEBUG: Predicted Intent: '{tag}' | Confidence: {confidence:.2f}")
    # ---------------------------

    # routing logic

    # path a - chatting
    if tag in ["greeting", "goodbye", "joke", "identity"] and confidence > 0.4:
        full_prompt = f"The user said: {user_input}. Reply in the character of a wise wizard."
        for chunk in llm.stream(full_prompt):
            print(chunk, end="", flush=True)
        print()
    # path b - dice
    elif tag == "roll_dice" and confidence > 0.4:
        print("Rolling dice...)")
        output = roll_dice(user_input)
        print(f"Bot: {output}")
    # path c - rules
    else:
        # Assume everything else is a D&D Question (RAG Path)
        full_prompt = query_dnd_manuals(user_input, category_filter="rules")
        print("Bot: ", end="", flush=True)
        for chunk in llm.stream(full_prompt):
            print(chunk, end="", flush=True)
        print()