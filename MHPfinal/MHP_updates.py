import gradio as gr
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
import os
from huggingface_hub import login
import pyttsx3  # For Text-to-Speech
import speech_recognition as sr  # For Speech-to-Text

# Hugging Face API Login
login("token")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "token"

chain = None  # Global variable for chatbot

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()

def process_pdf(file_paths):
    global chain
    if not file_paths:
        return "No files uploaded. Please upload PDF files."

    if not all(file_path.endswith('.pdf') for file_path in file_paths):
        return "Please upload valid PDF files."

    try:
        documents = []
        for file_path in file_paths:
            loader = PyMuPDFLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)

        if not documents:
            return "No text found in the uploaded PDFs."

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_text = text_splitter.split_documents(documents)

        if not split_text:
            return "No text found after splitting the documents."

        # Create vector store
        db = Chroma.from_documents(split_text, embeddings)

        # Initialize LLM
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3")

        # Create RetrievalQA chain
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

        return "Dataset processed successfully! You can now ask questions."
    except Exception as e:
        return f"Error processing PDFs: {str(e)}"

def answer_query(query):
    if chain is None:
        return "Please upload and process Dataset(Pdf's) first."
    
    try:
        result = chain.run(query)
        formatted_result = f"{query}\n {result.strip()}"
        
        # Speak the response
        speak_text(result.strip())

        return formatted_result
    except Exception as e:
        return f"Error during query: {str(e)}"

# 🗣️ Function: Convert Text to Speech (TTS)
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# 🎙️ Function: Convert Speech to Text (STT)

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    try:
        # Open the audio file
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)  # Convert to proper audio format
            text = recognizer.recognize_google(audio_data)  # Convert speech to text
            return text  # Return transcribed text

    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the speech."
    except sr.RequestError:
        return "Speech recognition service is unavailable."
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Mental Health Counselling Bot")
    gr.Markdown("Welcome to the Mental Health Counselling Assistant Bot! I can help you answer questions about your Mental Health.")

    chatbot = gr.Chatbot(label="Chat with the Assistant")
    msg = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    clear = gr.Button("Clear Chat")
    audio_input = gr.Microphone(label="🎤 Speak your question", type="filepath")  # ✅ Fixed microphone input
    audio_output = gr.Audio(label="🔊 Hear the response")

    # Function to handle chat interactions
    def respond(message, chat_history):
        if chain is None:
            chat_history.append((message, "Dataset's are still being processed. Please wait..."))
            return chat_history
        response = answer_query(message)
        chat_history.append((message, response))
        return chat_history

    # Function to handle voice input
    def respond_voice(audio, chat_history):
        if not audio:
            return chat_history.append((None, "No audio detected. Please try again."))
        
        text = speech_to_text(audio)  # ✅ Converts speech to text
        return respond(text, chat_history)  # ✅ Passes text to chatbot function

    # Function to clear chat
    def clear_chat():
        return []

    # Link Gradio events to functions
    msg.submit(respond, [msg, chatbot], chatbot)
    audio_input.change(respond_voice, [audio_input, chatbot], chatbot)  # ✅ Fixed microphone handling
    clear.click(clear_chat, None, chatbot, queue=False)

    # Automatically process PDFs when the interface loads
    def on_load():
        status = process_pdf(["combinedpdf_dataset.pdf"])
        return [(None, status)]  # Display status in chat
    
    demo.load(on_load, None, chatbot)

# Launch Gradio interface
demo.launch()

