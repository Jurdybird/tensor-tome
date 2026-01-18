import json
import numpy as np
import pickle
from typing import TypedDict, Literal, List

from tensorflow.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.graph import StateGraph, START, END

from dice_tool import roll_dice


# global setup & loading
print("Loading System...")

# load tf model & tokenizer for intent classification
model = load_model('chat_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as enc_file:
    lbl_encoder = pickle.load(enc_file)

max_len = 20

# load vector db
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="dnd_db", embedding_function=embedding_model)

# load llm
llm = ChatOllama(model="llama3.2", temperature=0.7)

# graph state definition
class AgentState(TypedDict):
    question: str
    intent: str
    confidence: float
    context: str
    answer: str

# node definitions

def classifier_node(state: AgentState):
    """
    Classifies the user's input using the pre-trained TensorFlow model.
    """
    user_input = state["question"]
    
    # TF Preprocessing
    seq = tokenizer.texts_to_sequences([user_input]) 
    padded = pad_sequences(seq, truncating='post', maxlen=max_len)
    
    # Prediction
    result = model.predict(padded, verbose=0)
    tag_index = np.argmax(result)
    tag = lbl_encoder.inverse_transform([tag_index])[0]
    confidence = result[0][tag_index]
    
    print(f"DEBUG: Intent='{tag}' | Confidence={confidence:.2f}")
    
    return {"intent": tag, "confidence": confidence}

def general_chat_node(state: AgentState):
    """
    Handles general chitchat (greeting, jokes, etc.)
    """
    prompt = PromptTemplate.from_template(
        "You are a wise D&D Wizard. The user said: {question}. Reply in character."
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": state["question"]})
    return {"answer": response}

def dice_node(state: AgentState):
    """
    Handles dice rolling requests.
    """
    result = roll_dice(state["question"])
    return {"answer": result}

def retrieve_node(state: AgentState):
    """
    Retrieves relevant documents from the Vector DB.
    """
    print("(Searching the rulebooks...)")
    # Search for top 5 matches, filter for 'rules' if applicable (or general search)
    # Using general search here for breadth
    results = db.similarity_search(state["question"], k=5)
    
    # Format context
    context_text = "\n\n".join([f"[Source: {doc.metadata.get('source_file', 'Unknown')}]\n{doc.page_content}" for doc in results])
    return {"context": context_text}

def rag_generate_node(state: AgentState):
    """
    Generates an answer using the retrieved context.
    """
    template = """You are a D&D Assistant. Answer the question using ONLY the context below.
    
    MANDATORY CITATION RULE:
    Every fact you state must be supported by the provided context.
    At the end of your answer, list the Source Files you used.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": response}

def fallback_node(state: AgentState):
    """
    Fallback if confidence is low.
    """
    return {"answer": "I'm not quite sure what you mean. Could you rephrase that in Common?"}


# conditional edges
def router(state: AgentState) -> Literal["general_chat", "dice_roll", "retrieve_rules", "fallback"]:
    intent = state["intent"]
    conf = state["confidence"]
    
    if conf < 0.4:
        return "fallback"
    
    if intent in ["greeting", "goodbye", "joke", "identity"]:
        return "general_chat"
    elif intent == "roll_dice":
        return "dice_roll"
    else:
        # Default to RAG for anything else (rules, monsters, spells)
        return "retrieve_rules"

# graph construction
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("dice_roll", dice_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rag_generate", rag_generate_node)
workflow.add_node("fallback", fallback_node)

workflow.add_edge(START, "classifier")
workflow.add_conditional_edges(
    "classifier",
    router,
    {
        "general_chat": "general_chat",
        "dice_roll": "dice_roll",
        "retrieve_rules": "retrieve",
        "fallback": "fallback"
    }
)

workflow.add_edge("retrieve", "rag_generate")

workflow.add_edge("general_chat", END)
workflow.add_edge("dice_roll", END)
workflow.add_edge("rag_generate", END)
workflow.add_edge("fallback", END)

app = workflow.compile()

# main loop
if __name__ == "__main__":
    print("Hybrid LangGraph Bot is ready! (Powered by Llama 3.2)")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Farewell, adventurer!")
                break
            
            # using invoke for simplicity
            initial_state = {"question": user_input, "intent": "", "confidence": 0.0, "context": "", "answer": ""}
            final_state = app.invoke(initial_state)
            
            print(f"Bot: {final_state['answer']}")
            
        except Exception as e:
            print(f"An error occurred: {e}")