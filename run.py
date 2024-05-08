from langchain_cohere import CohereEmbeddings
import streamlit as st
from PyPDF2 import PdfReader 

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

from htmlTemplates import css
from htmlTemplates import bot_template
from htmlTemplates import user_template
from dotenv import load_dotenv


# GET TEXT FROM PDF
def get_pdf_text(pdf_docs):
    text=""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text



# GET TEXT CHUNKS
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    
    return chunks



# EMBED THE TEXT AND MAKE IT VECTOR DB
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    # embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore
    

# CONVERSATION CHAIN
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# MAIN FUNCTION
def main():
    load_dotenv()

    # MAIN SECTION
    st.set_page_config(page_title='Chat with multiple pdfs', page_icon=':books:')
    
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    
    st.header('Chat with multiple PDFs :books: ')
    user_question = st.text_input('Ask a question about your document: ')

    if user_question:
        handle_userinput(user_question)

    # SIDEBAR SECTION
    with st.sidebar:
        st.subheader('Your documents')
        
        pdf_docs = st.file_uploader('Upload your pdfs', accept_multiple_files=True)
        
        if st.button('Process'):
            with st.spinner("Processing"):
                # STEP 1: GET THE PDF
                raw_text = get_pdf_text(pdf_docs)

                # STEP 2: GET THE TEXT CHUNKS FORM PDF
                text_chunks = get_text_chunks(raw_text)
                
                # STEP 3: CREATE VECTOR STORE
                vectorstore = get_vectorstore(text_chunks)
                
                # STEP 4 - CREATE CONVERSATION CHAIN
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()