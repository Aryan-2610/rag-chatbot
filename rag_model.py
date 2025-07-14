
 # File and system utilities
import os

# LangChain components
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import dotenv
# 🔍 Path to your file
filepath = "ALL_SCRAPED_TEXT.txt"

# 📦 File size in bytes
import os
file_size_bytes = os.path.getsize(filepath)
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024

# 🔢 Word count
with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()
    word_count = len(text.split())

# 📊 Results
print(f"📝 Word count: {word_count}")
print(f"📁 File size: {file_size_bytes} bytes ({file_size_kb:.2f} KB / {file_size_mb:.2f} MB)")
from langchain.document_loaders import TextLoader

# Your .txt file
text_path = "ALL_SCRAPED_TEXT.txt"

# Use LangChain’s loader
loader = TextLoader(text_path, encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)
print(f"📚 Split into {len(docs)} chunks.")

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Save vector DB to local
vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="rag_chroma_db"
)

vectorstore.persist()

from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()
# Set your API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load Gemini 1.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using **only** the information given in the context.

- Start with a natural, helpful tone — sound like you're speaking to a person.
- Use bullet points if the answer has multiple parts or levels.
- Be concise but don’t miss any detail available in the context.
- **Avoid phrases like "the context says" or "the text mentions"** — just give the answer.
- If the answer is not present in the context, respond politely with something like: "Sorry, I couldn’t find that information in the context."
-if it asks like list all or something like give details give answer in points
Context:
{context}

Question:
{question}

Answer:
""")


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your scraped text
text_path = "ALL_SCRAPED_TEXT.txt"
loader = TextLoader(text_path, encoding="utf-8")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)
print(f"✅ Split into {len(docs)} chunks.")

from langchain.chains import RetrievalQA

retriever = vectorstore.as_retriever(search_kwargs={"k"})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

def get_rag_response(query):
    result = qa_chain({"query": query})
    return result["result"]  # <- RETURN it, don’t print



get_rag_response("what is this page about")