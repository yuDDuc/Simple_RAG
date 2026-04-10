import os
from dotenv import load_dotenv

# Load PDF
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Split text
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google GenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Vector DB
from langchain_community.vectorstores import FAISS

# Chains
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Prompt
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables
print("🚀 Script starting...")
print("🔧 Reading .env and setting up API key...")
load_dotenv()

def setup_rag():
    print("🚀 Initializing RAG System...")
    
    # Check if folder exists
    if not os.path.exists("papers"):
        os.makedirs("papers")
        print("⚠️ 'papers/' directory created. Please add your PDFs there.")
    
    # 2. Load Papers
    print("Scanning 'papers/' directory for PDFs...")
    loader = DirectoryLoader("papers/", glob="*.pdf", loader_cls=PyPDFLoader)
    print("Parsing PDFs (this might take a minute depending on file size)...")
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

    if not docs:
        print("No PDF documents found in 'papers/' folder.")
        return None
        
    print(f"Loaded {len(docs)} pages.")

    # 3. Split Documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 4. Embedding + FAISS load/save
    print("Setting up embeddings + vector store...")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    faiss_index_exists = (
        os.path.exists("faiss_index/index.faiss") and
        os.path.exists("faiss_index/index.pkl")
    )
    if faiss_index_exists:
        print("Loading existing vector store...")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vector store...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local("faiss_index")

    print("Vector store ready.")
        
    # 5. Setup LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # 6. Setup Retrieval Chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. Answer in the same language as the question."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is present
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        print("Please set it in your environment or a .env file.")
        exit(1)

    chain = setup_rag()
    
    if chain:
        print("\nSYSTEM READY!")
        print("You can now ask questions about your papers (type 'exit' to quit).")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n Question: ")
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                
                if not query.strip():
                    continue

                print("Thinking...")
                response = chain.invoke({"input": query})
                
                print(f"\n Answer: {response['answer']}")
                print("-" * 50)
            except KeyboardInterrupt:
                print("\nGoodbye! ")
                break
            except Exception as e:
                print(f" An error occurred: {e}")
