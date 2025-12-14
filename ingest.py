import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/"
DB_PATH = "dnd_db"

# define how to categorize files based on names
def get_category(filename):
    filename = filename.lower()
    if "player" in filename:
        return "rules"
    elif "dungeon" in filename:
        return "rules"
    elif "monster" in filename:
        return "monsters"
    else:
        return "general"

documents = []
print("Scanning")
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)
        category = get_category(file)
        print(f"   - Loading: {file} --> Tagged as [{category}]")
        loader = PyPDFLoader(pdf_path)
        loaded_docs = loader.load()

        # add metadata to every page
        for doc in loaded_docs:
            doc.metadata["category"] = category
            doc.metadata["source_file"] = file # keeps book name

        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages total.")

# chunk the text to get snippets of rules
# chunk_size=1000: roughly a few paragraphs.
# chunk_overlap=200: keep some context between chunks so sentences dont get cut in half.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} text chunks.")

# create embeddings & save to db
# using standard, free model from huggingface to text -> math
print("Creating Database...")
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the chroma database and save it to disk
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_PATH
)

print("Database saved to '{DB_PATH}'.")