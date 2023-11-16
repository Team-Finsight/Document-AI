import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
import os
from dotenv import load_dotenv
import tempfile
from langchain.chat_models import ChatAnyscale
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io
from PIL import Image
import shutil
api_key = "esecret_fv9yhc2f1ix7lfdztdfh1fd6n8"
api_base = "https://api.endpoints.anyscale.com/v1"
pytesseract.pytesseract.tesseract_cmd = None

# search for tesseract binary in path
@st.cache_resource
def find_tesseract_binary() -> str:
    return shutil.which("tesseract")

# set tesseract binary path
pytesseract.pytesseract.tesseract_cmd = find_tesseract_binary()
if not pytesseract.pytesseract.tesseract_cmd:
    st.error("Tesseract binary not found in PATH. Please install Tesseract.")
class SimpleDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""

def ocr_pdf_to_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''

        # Create a temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                pix.save(image_path)

                # Perform OCR using the saved image path
                page_text = extract_text_from_image(image_path)
                text += page_text

        doc.close()
        return text
    except Exception as e:
        print(f"Error during OCR processing of PDF: {e}")
        return ""
    
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Create llm
    #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        #streaming=True, 
                        #callbacks=[StreamingStdOutCallbackHandler()],
                        #model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    llm = ChatAnyscale(anyscale_api_key=api_key, model_name="meta-llama/Llama-2-13b-chat-hf")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                    memory=memory)

    return chain

def main():
    load_dotenv()
    initialize_session_state()
    st.title("FIN-DOC CHAT🫡")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                extracted_text = loader.load()
                text.extend(extracted_text)
                
            elif file_extension in [".docx", ".doc"]:
                loader = Docx2txtLoader(temp_file_path)
                text.extend(loader.load())

            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
                text.extend(loader.load())

            elif file_extension in [".jpg", ".png", ".jpeg"]:
                extracted_text = extract_text_from_image(temp_file_path)
                simple_document = SimpleDocument(extracted_text)
                text.append(simple_document)
            else:
                st.warning(f"File format {file_extension} not supported.")

            

        # Text splitting and chunking
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        if not text_chunks:
            extracted_text1 = ocr_pdf_to_text(temp_file_path)
            simple_document = SimpleDocument(extracted_text1)
            text.append(simple_document)
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
            text_chunks = text_splitter.split_documents(text)
            print(text)
        os.remove(temp_file_path)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

        if not embeddings:
            st.error("Embedding generation failed. Please check your embedding model and input texts.")
            return

        try:
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        except IndexError as e:
            st.error(f"Failed to create the FAISS index: {e}")
            return
        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)

if __name__ == "__main__":
    main()
