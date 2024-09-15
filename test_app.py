import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_core.prompts import PromptTemplate
from prepare_vector_db import  *
from time import gmtime, strftime
import  os
import json
import os
import time

vector_db_path = r"vectorstores\db_faiss"
st.image(r"images/back_for_thanh_cong.png")
st.header("Hyundai AI Assistant")

def read_vectors_db():
    # Embeding

    embedding_model = OpenAIEmbeddings(openai_api_key= "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")
    db = FAISS.load_local(vector_db_path, embedding_model,allow_dangerous_deserialization=True)
    return db



def get_conversation_chain(vectorstore):
    """
    Save the history of chat in a conversation chain
    :param vectorstore:
    :return:
    """
    llm = ChatOpenAI(openai_api_key= "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",
                     model="gpt-4o-mini")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),

        memory=memory
    )
    return conversation_chain

def response_generator(text):

    for word in text.strip():
        yield word + ""
        time.sleep(0.01)
def handle_userinput(user_question,template):
    st.chat_message("user", avatar="images/for_user.jpg").markdown(user_question)
    response = st.session_state.conversation({'question': template})
    st.session_state.chat_history = response['chat_history']
    save_chat_history(st.session_state.chat_history)  # Save chat history

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:

            st.session_state.messages.append(
                {"role": "user", "content": user_question, "avatar": "images/for_user.jpg"})
        else:
            with st.chat_message("assistant", avatar="images/logo_thanh_cong.jpg"):

                response = st.write_stream(response_generator(message.content))
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "avatar": "images/logo_thanh_cong.jpg"})

def save_chat_history(chat_history):
    history = [{'role': 'user' if i % 2 == 0 else 'bot', 'content': message.content} for i, message in
               enumerate(chat_history)]
    with open('chat_history.json', 'w') as f:
        json.dump(history, f, indent=4)

def response_generator(text):

    for word in text.strip():
        yield word + ""
        time.sleep(0.01)
def upload_file(uploaded_files):
    UPLOAD_DIR = "data"
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # Check if any files were uploaded
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Create a path to save the file locally
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            # Save the uploaded file to the directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Update knowledge successfully")
            create_db_from_files(UPLOAD_DIR)
    else:
        st.info("Please upload one or more files.")
def main():
    load_dotenv()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None



    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])



    prompt_template = PromptTemplate.from_template(
        """Bạn là một AI hỗ trợ sale thông minh của Công ty Công ty cổ phần liên doanh ô tô Hyundai Thành Công Việt Nam, nhiệm vụ của bạn là sẽ giúp đỡ tôi là một nhân viên tư vấn cho Hyundai Thành công trong việc tư vấn xe cho khách hàng. hãy cư sử chuẩn mực và suy nghĩ chính xác trước khi đưa ra câu trả lời.
        Nếu bạn không biết câu trả lời hãy hãy nói không biết, đừng cố tạo ra câu trả lời
        Tôi sẽ có thể nhắc rõ bạn mọi lần nhưng hãy đừng coi đó là điều lạ hãy trả lời như tôi đã nói với bạn trước
        Nếu có thể bạn hãy đưa thêm kết quả mỗi lần tìm được và đưa cho tôi một thông tin gì đó hay để đưa ra cho khách hàng
        Hãy trả lời theo như file tôi đã cung cấp cho bạn và câu hỏi đó chính là {question}
        Và không cần tỏ ra quá trang trọng và chào hỏi trong các lần hỏi"""
    )


    if prompt := st.chat_input("What is up?"):

        handle_userinput(prompt,prompt_template.format(question = prompt))

    vectorstore = read_vectors_db()

    st.session_state.conversation = get_conversation_chain(vectorstore)
    with st.sidebar:
        st.subheader("Update knowledge")
        uploaded_pdf = st.file_uploader("Upload a PDF", type=None,accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text


                # Create the database from the uploaded PDF
                if uploaded_pdf:
                    upload_file(uploaded_pdf)
                    st.success("Database created and saved locally.")

        if st.button("Save Conversation"):
            if st.session_state.chat_history:
                save_chat_history(st.session_state.chat_history)
                st.success("Conversation saved successfully!")
            else:
                st.warning("No conversation history to save.")

if __name__ == "__main__":
    main()
# st.write(st.session_state.messages)
# st.write(st.session_state.chat_history)
# st.write(st.session_state.conversation)