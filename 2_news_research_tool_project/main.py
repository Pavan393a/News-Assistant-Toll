import os
import re
import shutil
import streamlit as st
import pandas as pd
from datetime import datetime
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------- Environment Setup --------------------
load_dotenv()
st.set_page_config(page_title="RockyBot: News Research Tool", layout="wide")


# -------------------- Helper Functions --------------------
def extract_urls_from_text(text):
    """Extract URLs from text input."""
    if not text.strip():
        return []
    urls = re.findall(r'https?://[^\s]+', text)
    return urls or text.splitlines()


def send_email(subject, body, to_email):
    """Send email using SMTP."""
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv("GMAIL_USER")
        sender_password = os.getenv("GMAIL_PASSWORD")

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False


def process_urls(urls, file_path, embeddings):
    """Process URLs and create FAISS index."""
    if not urls:
        st.warning("Please enter at least one valid URL or upload a file.")
        return False

    # Remove existing FAISS index
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    with st.spinner("🔄 Loading and processing data... Please wait..."):
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Store processed data
            st.session_state.processed_data.extend(data)

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            # Create FAISS index and save locally
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            vectorstore_openai.save_local(file_path)

            # Update URLs in session state
            st.session_state.urls.extend(urls)

            st.success("✅ URLs processed and embeddings saved successfully!")
            return True

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return False


# -------------------- Initialization --------------------
file_path = "faiss_store_openai"
llm = OpenAI(temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings()

# Initialize session state
if "user_data" not in st.session_state:
    st.session_state.user_data = {"nickname": "", "role": "", "login_time": datetime.now()}
    st.session_state.admin_data = []
    st.session_state.messages = []
    st.session_state.urls = []
    st.session_state.processed_data = []
    st.session_state.scheduled_urls = []

# -------------------- Main Application --------------------
tabs = st.radio("Select a page:", ["Home", "Admin"])

# -------------------- Home Page --------------------
if tabs == "Home":
    if not st.session_state.user_data["nickname"]:
        st.title("Welcome to RockyBot!")
        nickname = st.text_input("Enter your Nickname:")
        role = st.selectbox("Select your Role:", ["Guest", "Researcher", "Admin"])

        if st.button("Submit"):
            if nickname and role:
                st.session_state.user_data.update({
                    "nickname": nickname,
                    "role": role,
                    "login_time": datetime.now()
                })
                st.session_state.admin_data.append({
                    "nickname": nickname,
                    "role": role,
                    "login_time": datetime.now()
                })
            else:
                st.warning("Please enter your nickname and role.")

    else:
        st.title("RockyBot: News Research Tool 📈")

        # -------------------- URL Input Section --------------------
        st.sidebar.subheader("Enter News Article URLs")

        # Manual URL input
        manual_urls = st.sidebar.text_area(
            "Enter URLs (one per line)",
            placeholder="https://example.com\nhttps://another-example.com"
        )

        # File upload
        uploaded_file = st.sidebar.file_uploader("Or upload a text file with URLs", type=["txt"])

        # Process button
        if st.sidebar.button("Process URLs"):
            all_urls = []

            # Process manual URLs
            if manual_urls:
                manual_urls_list = extract_urls_from_text(manual_urls)
                all_urls.extend(manual_urls_list)

            # Process uploaded file
            if uploaded_file:
                try:
                    file_content = uploaded_file.read().decode("utf-8")
                    file_urls = extract_urls_from_text(file_content)
                    all_urls.extend(file_urls)
                except Exception as e:
                    st.error(f"Error reading uploaded file: {str(e)}")

            # Remove duplicates while preserving order
            all_urls = list(dict.fromkeys(all_urls))

            if all_urls:
                success = process_urls(all_urls, file_path, embeddings)
                if success:
                    st.sidebar.success("Ready to answer questions about the processed URLs!")

        # -------------------- News Scheduler --------------------
        with st.sidebar.expander("News Scheduler ⏰", expanded=False):
            st.write("Schedule daily or weekly automatic news retrieval.")

            schedule_time = st.time_input("Select Time for News Retrieval")
            frequency = st.selectbox("Select Frequency:", ["Daily", "Weekly"])
            email = st.text_input("Enter your Email:")
            scheduled_url = st.text_area("Enter URL(s) to Schedule:")
            notification_format = st.selectbox(
                "Select Notification Format:",
                ["Key Points", "Context", "Summary"]
            )

            if st.button("Schedule Retrieval"):
                if email and scheduled_url:
                    scheduled_details = {
                        "time": schedule_time,
                        "frequency": frequency,
                        "email": email,
                        "urls": scheduled_url.splitlines(),
                        "notification_format": notification_format
                    }
                    st.session_state.scheduled_urls.append(scheduled_details)

                    # Prepare email content
                    nickname = st.session_state.user_data["nickname"]
                    email_subject = f"News Scheduler Confirmation for {nickname}"
                    email_content = f"""
                    <html>
                    <body>
                        <h2>Hello {nickname},</h2>
                        <p>Your news retrieval has been scheduled successfully with the following details:</p>
                        <ul>
                            <li><strong>Scheduled Time:</strong> {schedule_time}</li>
                            <li><strong>Frequency:</strong> {frequency}</li>
                            <li><strong>Notification Format:</strong> {notification_format}</li>
                            <li><strong>URLs:</strong></li>
                            <ul>
                                {''.join(f'<li>{url}</li>' for url in scheduled_details['urls'])}
                            </ul>
                        </ul>
                        <p>You will receive updates at this email address in the selected format.</p>
                        <p>Thank you for using RockyBot!</p>
                    </body>
                    </html>
                    """

                    if send_email(email_subject, email_content, email):
                        st.success(f"News Retrieval scheduled at {schedule_time} ({frequency})")
                else:
                    st.warning("Please provide your email and URL(s).")

        # -------------------- Chat Interface --------------------
        if os.path.exists(file_path):
            st.subheader("Ask Questions About the News")

            # Initialize FAISS index and retriever
            vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever()

            # Display Chat History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User Input
            if user_input := st.chat_input("Ask a question about the news articles..."):
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.chat_message("user"):
                    st.markdown(user_input)

                # Generate answer
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Get the base result
                        result = chain(
                            {
                                "question": user_input + " Please provide a detailed response that I can later format into concise, detailed, and summary versions."
                            },
                            return_only_outputs=True
                        )
                        full_answer = result.get("answer", "No answer available.")
                        sources = result.get("sources", "")

                        # Create distinct versions of the answer
                        # Concise: Key points and main takeaway
                        concise_prompt = user_input + " Please provide a very concise answer focusing only on the key points."
                        concise_result = chain({"question": concise_prompt}, return_only_outputs=True)
                        concise_answer = concise_result.get("answer", "No answer available.")

                        # Summary: Overview with context
                        summary_prompt = user_input + " Please provide a brief summary that gives context and main conclusions."
                        summary_result = chain({"question": summary_prompt}, return_only_outputs=True)
                        summary_answer = summary_result.get("answer", "No answer available.")

                        # Store the detailed answer in chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_answer})

                        # Display answer versions
                        tabs = st.tabs(["Concise", "Detailed", "Summary"])

                        with tabs[0]:
                            st.write("**Quick Answer:**")
                            st.write(concise_answer)
                        with tabs[1]:
                            st.write("**Full Analysis:**")
                            st.write(full_answer)
                        with tabs[2]:
                            st.write("**Executive Summary:**")
                            st.write(summary_answer)

                        # Display sources
                        if sources:
                            st.subheader("Sources:")
                            for source in sources.split("\n"):
                                if source.strip():
                                    st.markdown(f"- {source.strip()}")

                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

            # -------------------- Session Metrics --------------------
            st.sidebar.subheader("Session Information")
            if st.session_state.messages:
                question_count = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
                st.sidebar.write(f"Total Questions Asked: {question_count}")

                # Chat History Download
                if st.sidebar.button("Download Chat History"):
                    chat_df = pd.DataFrame(st.session_state.messages)
                    st.sidebar.download_button(
                        "Download as CSV",
                        chat_df.to_csv(index=False).encode("utf-8"),
                        "chat_history.csv",
                        "text/csv"
                    )

# -------------------- Admin Page --------------------
elif tabs == "Admin":
    st.title("Admin Panel")
    secret_code = st.text_input("Enter Admin Secret Code:", type="password")

    if secret_code == "password":  # Replace with secure authentication
        st.subheader("User Management")

        # Display Active Users
        st.subheader("Active Users")
        if st.session_state.admin_data:
            df = pd.DataFrame(st.session_state.admin_data)
            st.dataframe(df)
        else:
            st.write("No active users currently tracked.")

        # Display Scheduled News Retrievals
        st.subheader("Scheduled News Retrieval")
        if st.session_state.scheduled_urls:
            scheduled_df = pd.DataFrame(st.session_state.scheduled_urls)
            st.dataframe(scheduled_df)
        else:
            st.write("No scheduled news retrievals.")
    else:
        st.warning("Please enter the correct admin secret code.")

if __name__ == "__main__":
    # You can add any startup configurations here
    pass