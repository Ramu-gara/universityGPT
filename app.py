from flask import Flask, request, jsonify, render_template, session
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import os
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from langchain.docstore.document import Document
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader


app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "your_secret_key"  # For session handling

# Store user memory
user_memory = defaultdict(lambda: {"name": None, "questions": []})

def load_documents():
    # URLs to load from
    urls = [
        'https://www.csueastbay.edu/msba/',
        'https://www.csueastbay.edu/msba/howtoapply.html',
        'https://www.csueastbay.edu/msba/academics.html',
        'https://www.csueastbay.edu/msba/careers.html',
        'https://www.csueastbay.edu/msba/infosessions.html',
        'https://www.csueastbay.edu/msba/faq.html',
        'https://www.csueastbay.edu/msba/academics.html',
        'https://www.csueastbay.edu/directory/profiles/mgmt/kiminkyu.html',
        'https://www.csueastbay.edu/clubsandorgs/rso-advisor-info/active-clubs-organizations.html'
        # Add more URLs as needed
    ]
    
    # Load documents from web
    loader = WebBaseLoader(urls)
    web_documents = loader.load()

    # Load and process data.pkl
    with open('data.pkl', 'rb') as file:
        pkl_data = pickle.load(file)
    
    # Convert the pickle data into text format
    if isinstance(pkl_data, dict):
        pkl_text = "\n".join([f"{key}: {value}" for key, value in pkl_data.items()])
    elif isinstance(pkl_data, list):
        pkl_text = "\n".join(map(str, pkl_data))
    else:
        pkl_text = str(pkl_data)
    
    # Wrap pickle text into a Document format
    pkl_documents = [Document(page_content=pkl_text, metadata={"source": "data.pkl"})]
    
    # Combine web and pickle documents
    all_documents = web_documents + pkl_documents
    
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(all_documents)

# Initialize embeddings
def initialize_embeddings():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

# Initialize FAISS vector store
def initialize_vectorstore(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)

# Initialize LLM and Retrieval Chain
def initialize_llm_and_chain(vectorstore):
    llm = ChatGroq(
        temperature=0,
        groq_api_key='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',
        model_name="llama-3.1-70b-versatile"
    )
    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    CONTEXT: {context}
    QUESTION: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

# Initialize global variables
documents = load_documents()
embedding_model = initialize_embeddings()
vectorstore = initialize_vectorstore(documents, embedding_model)
retrieval_chain = initialize_llm_and_chain(vectorstore)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    # Function to extract and remember user name
    def remember_name(query_text):
        if "i'm" in query_text.lower() or "i am" in query_text.lower():
            name = query_text.split("i'm", 1)[-1].strip() if "i'm" in query_text.lower() else query_text.split("i am", 1)[-1].strip()
            return name.split()[0]  # Return the first word as the name
        return None

    # Retrieve session ID and initialize memory for this session
    user_id = session.get('user_id', request.remote_addr)
    memory = user_memory[user_id]

    # Retrieve the query from the request
    data = request.json
    query_text = data.get('query', "").strip()

    # Remember the name if provided in the query
    name = remember_name(query_text)
    if name:
        memory['name'] = name

    # Store the question in memory
    memory['questions'].append(query_text)

    # Check for basic responses
    if memory['name']:
        greeting_responses = {
            "hi": f"Hello {memory['name']}! How can I assist you today?",
            "hello": f"Hi {memory['name']}! What would you like to know?",
        }
    else:
        greeting_responses = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hi there! What would you like to know?",
            
        }

    basic_response = greeting_responses.get(query_text.lower(), None)
    if basic_response:
        return jsonify({
            "result": basic_response,
            "source_documents": []
        })

    # Use the retrieval chain for complex queries
    response = retrieval_chain.invoke({"query": query_text})
    return jsonify({
        "result": response["result"],
       # "source_documents": [doc.page_content for doc in response["source_documents"]],
        "user_memory": memory  # Debugging purpose: to check memory contents
    })

if __name__ == '__main__':
    app.run(debug=True)
