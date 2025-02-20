import gradio as gr
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
import os
from huggingface_hub import login

login("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")


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

chain = None

def process_pdf(file_paths):
    global chain
    if not file_paths:
        return "No files uploaded. Please upload PDF files."
    
    if not all(file_path.endswith('.pdf') for file_path in file_paths):
        return "Please upload valid PDF files."
    
    try:
        # Load and process the PDFs
        documents = []
        for file_path in file_paths:
            try:
                print(f"Processing file: {file_path}")  # Debugging
                loader = PyMuPDFLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} documents from {file_path}")  # Debugging
            except Exception as e:
                print(f"Error loading {file_path}: {e}")  # Debugging
                return f"Error loading {file_path}: {str(e)}"
        
        print(f"Total documents loaded: {len(documents)}")  # Debugging
        
        if not documents:
            return "No text found in the uploaded PDFs."
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("Embeddings created successfully.")  # Debugging
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_text = text_splitter.split_documents(documents)
        print(f"Text after splitting: {split_text[:2]}")  # Debugging
        
        if not split_text:
            return "No text found after splitting the documents."
        
        # Create vector store
        db = Chroma.from_documents(split_text, embeddings)
        print(f"Chroma database contains {len(db)} vectors.")  # Debugging
        
        # Initialize LLM
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3")  # Alternative LLM for QA
        print("LLM initialized successfully.")  # Debugging
        
        # Create RetrievalQA chain
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
        print("QA chain initialized successfully.")  # Debugging
        
        return "Dataset processed successfully! You can now ask questions."
    except Exception as e:
        print(f"Error processing PDFs: {e}")  # Debugging
        return f"Error processing PDFs: {str(e)}"

def answer_query(query):
    if chain is None:
        return "Please upload and process Dataset(Pdf's) first."
    
    try:
        result = chain.run(query)
        
        # Extract only the relevant answer and format it properly
        if "{" in result:  # Check if response contains unwanted JSON-like text
            start_idx = result.find("Helpful Answer:")
            if start_idx != -1:
                result = result[start_idx:]  # Keep only the answer part
        
        formatted_result = f"{query}\n {result.strip()}"
        
        print(f"Query result: {formatted_result}")  # Debugging
        return formatted_result
    
    except Exception as e:
        print(f"Error during query: {e}")  # Debugging
        return f"Error during query: {str(e)}"


# Function to programmatically upload and process PDFs
def programmatic_upload_and_process():
    # List of PDF file paths to upload programmatically
    pdf_paths = [
        "combinedpdf_dataset.pdf", 
        "Bhagavad_Gita_As_It_Is.pdf",
        # "path/to/your/file3.pdf"
    ]
    
    try:
        # Call the process_pdf function directly with the file paths
        status = process_pdf(pdf_paths)
        
        # Ensure the chain is initialized
        global chain
        if chain is None:
            raise Exception("Failed to initialize the QA chain.")
            
        return "Dataset processed successfully! You can now ask questions."
    
    except Exception as e:
        print(f"Error during programmatic upload: {e}")
        return f"Error: {str(e)}"
    
# Gradio Interface (Assistant Bot Style)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Title and description
    gr.Markdown("# 🤖 Mental Health Counselling Bot")
    gr.Markdown("Welcome to the Mental Health Counselling Assistant Bot! I can help you answer questions about your Mental Health.")
    
    # Chatbot interface
    chatbot = gr.Chatbot(label="Chat with the Assistant")
    msg = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    clear = gr.Button("Clear Chat")
    
    # Function to handle chat interactions
    def respond(message, chat_history):
        if chain is None:
            chat_history.append((message, "Dataset's are still being processed. Please wait..."))
            return chat_history
        response = answer_query(message)
        chat_history.append((message, response))
        return chat_history
    
    # Function to clear the chat
    def clear_chat():
        return []
    
    # Link Gradio events to functions
    msg.submit(respond, [msg, chatbot], chatbot)
    clear.click(clear_chat, None, chatbot, queue=False)
    
    # Automatically process PDFs when the interface loads
    def on_load():
        status = programmatic_upload_and_process()
        return [(None, status)]  # Display status in chat
    
    demo.load(on_load, None, chatbot)

# Launch the Gradio interface
demo.launch()