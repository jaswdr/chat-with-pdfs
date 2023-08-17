import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

def check_dependencies():
    # Confirm that PyTorch is installed and CUDA is available
    import torch
    assert torch.cuda.is_available(), 'CUDA is not available.'

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.getvalue()))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vector_store

def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_pdfs(pdf_docs):
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    with st.spinner('Processing PDFs...'):
        # get pdf text
        raw_text = get_pdf_text(pdf_docs)

        # get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # create vector store
        vector_store = get_vector_store(text_chunks)
        
        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vector_store)
        st.success('PDFs processed successfully!')

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            with st.chat_message(name='user', avatar='user'):
                st.write(message.content)
        else:
            with st.chat_message(name='assistant', avatar='assistant'):
                st.write(message.content)

def build_ui():
    st.set_page_config(page_title='Chat with Multiple PDFs', page_icon=':books:', layout='wide')
    st.header('Chat with multiple PDFs :books:')
    user_question = st.chat_input('Ask a question about your documents:')
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader('Upload PDFs')
        pdf_docs = st.file_uploader('Upload PDFs here and click on "Process":', type=['pdf'], accept_multiple_files=True)
        if st.button('Process'):
            process_pdfs(pdf_docs)

def main():
    load_dotenv()
    check_dependencies()
    build_ui()

if __name__ == '__main__':
    main()