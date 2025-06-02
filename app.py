from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader, BSHTMLLoader
from langchain.chains.question_answering.chain import load_qa_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
load_dotenv()
import google.generativeai as genai
import re
from langchain_core.chat_history import InMemoryChatMessageHistory
from datetime import datetime, timedelta
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gradio as gr
import tempfile
import shutil

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# local timezone
local_tz = pytz.timezone('Asia/Kathmandu')

# Global variables
vector_store = None
llm = None
qa_chain = None
chain_with_history = None
chat_history = None

# State variables
COLLECTING_INFO_STATE = None
APPOINTMENT_STATE = None
USER_INFO = {}
APPOINTMENT_INFO = {}
NEXT_ACTION = None

# Keywords for different intents
CALL_REQUEST_KEYWORDS = ["call me", "callback", "contact me", "reach out", "phone me"]
APPOINTMENT_KEYWORDS = ["book appointment", "schedule appointment", "make appointment", 
                       "book meeting", "schedule meeting", "appointment", "meeting"]

def initialize_llm():
    """Initialize the LLM and create the QA chain"""
    global llm, qa_chain, chain_with_history, chat_history
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    chat_history = InMemoryChatMessageHistory()
    
    prompt_template = """You are a helpful assistant.

Conversation History:
{chat_history}

Context (Document Content):
{context}

Question: {input}

Instructions:
1. When summarizing the document or providing its gist, use ONLY the information provided in the 'Context (Document Content)' section. Do NOT include any personal details (like names, phone numbers, or emails) that might appear in the 'Conversation History' unless explicitly asked about them in relation to the document content.
2. For all other questions, answer based on both the 'Context (Document Content)' and the 'Conversation History' to maintain conversational flow and answer directly.
3. Be concise and to the point.

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template,
                          input_variables=["context", "chat_history", "input"])
    
    qa_chain = PROMPT | llm
    
    chain_with_history = RunnableWithMessageHistory(
        qa_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

def load_document(file_path):
    """Load and process a document into the vector store"""
    global vector_store
    
    try:
        # Determine the appropriate loader based on file extension
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        elif file_path.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_path.endswith(('.html', '.htm')):
            loader = BSHTMLLoader(file_path)
        else:
            return False, "Unsupported document format. Please use .txt, .pdf, .csv, .md, or .html files."
        
        # Load documents
        documents = loader.load()
        if not documents:
            return False, "Could not load any documents. Please check the file format."
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        if not texts:
            return False, "Could not split documents into text chunks."
        
        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vector_store = FAISS.from_documents(texts, embeddings)
        
        return True, f"Successfully loaded {len(documents)} documents and created {len(texts)} text chunks."
    
    except Exception as e:
        return False, f"Error loading document: {str(e)}"

def upload_document(file):
    """Handle document upload from Gradio"""
    if file is None:
        return "No file uploaded.", ""
    
    try:
        # Create a temporary file with the uploaded content
        temp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
            temp_path = temp_file.name
            shutil.copy2(file.name, temp_path)
        
        # Load the document
        success, message = load_document(temp_path)
        
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        if success:
            return f"{message}", "Document loaded successfully! You can now ask questions about it."
        else:
            return f"{message}", ""
    
    except Exception as e:
        return f"Error uploading file: {str(e)}", ""

def is_call_request(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in CALL_REQUEST_KEYWORDS)

def is_appointment_request(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in APPOINTMENT_KEYWORDS)

def extract_date_llm(user_input):
    """Use LLM to extract and convert dates to YYYY-MM-DD format"""
    today_local = datetime.now(local_tz).strftime('%Y-%m-%d')
    day_of_week = datetime.now(local_tz).strftime('%A')
    
    prompt = f"""Today is {today_local} ({day_of_week}). 
    Convert this date expression to ISO date format (YYYY-MM-DD): '{user_input}'. 
    
    Examples:
    - "tomorrow" → {(datetime.now(local_tz) + timedelta(days=1)).strftime('%Y-%m-%d')}
    - "next Monday" → (calculate the next Monday's date)
    - "2024-12-25" → 2024-12-25
    - "December 25" → 2024-12-25
    
    Only return the date in YYYY-MM-DD format. Do not include any explanation or additional text."""
    
    try:
        response = llm.invoke(prompt)
        extracted_date = response.content.strip()
        
        # Validate the extracted date format
        datetime.strptime(extracted_date, '%Y-%m-%d')
        
        # Check if the date is not in the past
        extracted_datetime = datetime.strptime(extracted_date, '%Y-%m-%d')
        today = datetime.now(local_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        
        if extracted_datetime.date() < today.date():
            return None, "The date you specified appears to be in the past. Please provide a future date."
        
        return extracted_date, None
        
    except ValueError:
        return None, "I couldn't understand that date format. Please try again."
    except Exception as e:
        return None, "There was an error processing the date. Please try again."

def send_appointment_email(user_info, appointment_info):
    """Send appointment confirmation email"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = user_info['email']
        msg['Subject'] = "Appointment Confirmation"
        
        body = f"""
        Dear {user_info['name']},
        
        Your appointment has been successfully booked!
        
        Details:
        - Purpose: {appointment_info['purpose']}
        - Date: {appointment_info['date']}
        - Name: {user_info['name']}
        - Phone: {user_info['phone']}
        - Email: {user_info['email']}
        
        We will contact you shortly to confirm the time.
        
        Best regards,
        Chatbot Assistant
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, user_info['email'], text)
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

def handle_information_collection(query):
    global COLLECTING_INFO_STATE, USER_INFO, NEXT_ACTION, APPOINTMENT_STATE

    if COLLECTING_INFO_STATE == "NAME":
        USER_INFO["name"] = query
        COLLECTING_INFO_STATE = "PHONE"
        return f"Thank you, {query}. What is your phone number?"
    
    elif COLLECTING_INFO_STATE == "PHONE":
        phone_regex = r"^(\+977\s*)?0?\d{10}$"
        if re.fullmatch(phone_regex, query.strip()) and sum(c.isdigit() for c in query) >= 9:
            USER_INFO["phone"] = query.strip()
            COLLECTING_INFO_STATE = "EMAIL"
            return "Great! And finally, what is your email address?"
        else:
            return "Please provide a valid phone number.\
                eg:- 9876543210"
    
    elif COLLECTING_INFO_STATE == "EMAIL":
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if re.fullmatch(email_regex, query.strip()):
            USER_INFO["email"] = query.strip()
            COLLECTING_INFO_STATE = None
            
            # Store user info in chat history
            user_info_message = f"My name is {USER_INFO['name']}, phone is {USER_INFO['phone']}, and email is {USER_INFO['email']}."
            chat_history.add_user_message(user_info_message)
            
            # Check what to do next based on NEXT_ACTION
            if NEXT_ACTION == "APPOINTMENT":
                APPOINTMENT_STATE = "PURPOSE"
                NEXT_ACTION = None
                return (f"Thank you, {USER_INFO.get('name', 'User')}! I have your contact details. "
                       f"Now, what is the purpose of your appointment?")
            elif NEXT_ACTION == "CALLBACK":
                NEXT_ACTION = None
                return (f"Thank you, {USER_INFO.get('name', 'User')}! We have your details:\n"
                       f"Phone: {USER_INFO.get('phone')}\n"
                       f"Email: {USER_INFO.get('email')}\n"
                       "What else can I help you with?")
            else:
                return (f"Thank you, {USER_INFO.get('name', 'User')}! We have your details:\n"
                       f"Phone: {USER_INFO.get('phone')}\n"
                       f"Email: {USER_INFO.get('email')}\n"
                       "Your information has been saved. What else can I help you with?")
        else:
            return "That doesn't look like a valid email address. Please enter a valid email address (e.g., user@example.com)."
    
    return "I'm not sure. Could you please clarify?"

def handle_appointment_booking(query):
    global APPOINTMENT_STATE, APPOINTMENT_INFO, USER_INFO, COLLECTING_INFO_STATE, NEXT_ACTION

    if not APPOINTMENT_STATE:
        if USER_INFO and USER_INFO.get('name') and USER_INFO.get('phone') and USER_INFO.get('email'):
            APPOINTMENT_STATE = "AWAITING_INFO_CHOICE"
            return (f"I already have your details as:\n"
                    f"Name: {USER_INFO.get('name')}\n"
                    f"Phone: {USER_INFO.get('phone')}\n"
                    f"Email: {USER_INFO.get('email')}\n"
                    "Would you like to use this information to book the appointment, or enter new details?\n"
                    "Please reply with 'use previous' or 'new details'.")
        else:
            COLLECTING_INFO_STATE = "NAME"
            NEXT_ACTION = "APPOINTMENT"
            return "Sure! Let's start by collecting your details. What is your full name?"

    if APPOINTMENT_STATE == "AWAITING_INFO_CHOICE":
        lower_query = query.lower()
        if "use previous" in lower_query:
            APPOINTMENT_STATE = "PURPOSE"
            return "Great! What is the purpose of your appointment?"
        elif "new" in lower_query:
            USER_INFO.clear()
            COLLECTING_INFO_STATE = "NAME"
            APPOINTMENT_STATE = None
            NEXT_ACTION = "APPOINTMENT"
            return "Alright, let's start with your new details. What is your full name?"
        else:
            return "Please reply with either 'use previous' or 'new details'."

    if APPOINTMENT_STATE == "PURPOSE":
        APPOINTMENT_INFO["purpose"] = query
        APPOINTMENT_STATE = "DATE"
        return "Got it. What date would you prefer for the appointment?"

    elif APPOINTMENT_STATE == "DATE":
        date_str, error = extract_date_llm(query)
        if error:
            return error
        APPOINTMENT_INFO["date"] = date_str
        APPOINTMENT_STATE = "VERIFY"
        
        # Show all details for verification
        return (f"Perfect! Let me verify your appointment details:\n\n"
                f"📋 **Appointment Summary:**\n"
                f"• **Name:** {USER_INFO['name']}\n"
                f"• **Phone:** {USER_INFO['phone']}\n"
                f"• **Email:** {USER_INFO['email']}\n"
                f"• **Purpose:** {APPOINTMENT_INFO['purpose']}\n"
                f"• **Date:** {APPOINTMENT_INFO['date']}\n\n"
                f"Please review the above details carefully. If everything looks correct, type **'confirm'** to book your appointment.\n"
                f"If you need to make changes, type **'change'** and I'll help you update the information.")

    elif APPOINTMENT_STATE == "VERIFY":
        lower_query = query.lower().strip()
        
        if lower_query == "confirm":
            APPOINTMENT_STATE = None
            if send_appointment_email(USER_INFO, APPOINTMENT_INFO):
                return (f"🎉 **Appointment Confirmed!**\n\n"
                        f"Your appointment for **{APPOINTMENT_INFO['purpose']}** on **{APPOINTMENT_INFO['date']}** "
                        f"has been booked successfully!\n\n"
                        f"📧 A confirmation email has been sent to **{USER_INFO['email']}**.\n"
                        f"📞 We will contact you shortly to confirm the exact time.\n\n"
                        f"Thank you for booking with us! Is there anything else I can help you with?")
            else:
                return (f"✅ Your appointment for **{APPOINTMENT_INFO['purpose']}** on **{APPOINTMENT_INFO['date']}** "
                        f"has been booked successfully!\n\n"
                        f"⚠️ However, there was an issue sending the confirmation email. "
                        f"Please note down your appointment details.\n\n"
                        f"Is there anything else I can help you with?")
        
        elif lower_query == "change":
            APPOINTMENT_STATE = "CHANGE_CHOICE"
            return (f"Sure! What would you like to change?\n\n"
                    f"Please choose from the following options:\n"
                    f"• Type **'name'** - to change your name\n"
                    f"• Type **'phone'** - to change your phone number\n"
                    f"• Type **'email'** - to change your email address\n"
                    f"• Type **'purpose'** - to change the appointment purpose\n"
                    f"• Type **'date'** - to change the appointment date\n"
                    f"• Type **'all'** - to start over with all details")
        
        else:
            return (f"I didn't understand that. Please type either:\n"
                    f"• **'confirm'** - to book the appointment with the details shown above\n"
                    f"• **'change'** - to modify any of the information")

    elif APPOINTMENT_STATE == "CHANGE_CHOICE":
        lower_query = query.lower().strip()
        
        if lower_query == "name":
            COLLECTING_INFO_STATE = "NAME"
            APPOINTMENT_STATE = None
            NEXT_ACTION = "APPOINTMENT"
            return "Please enter your new name:"
        
        elif lower_query == "phone":
            COLLECTING_INFO_STATE = "PHONE"
            APPOINTMENT_STATE = None
            NEXT_ACTION = "APPOINTMENT"
            return "Please enter your new phone number:"
        
        elif lower_query == "email":
            COLLECTING_INFO_STATE = "EMAIL"
            APPOINTMENT_STATE = None
            NEXT_ACTION = "APPOINTMENT"
            return "Please enter your new email address:"
        
        elif lower_query == "purpose":
            APPOINTMENT_STATE = "PURPOSE"
            return "Please enter the new purpose for your appointment:"
        
        elif lower_query == "date":
            APPOINTMENT_STATE = "DATE"
            return "Please enter your new preferred date for the appointment:"
        
        elif lower_query == "all":
            USER_INFO.clear()
            APPOINTMENT_INFO.clear()
            COLLECTING_INFO_STATE = "NAME"
            APPOINTMENT_STATE = None
            NEXT_ACTION = "APPOINTMENT"
            return "Let's start fresh! What is your full name?"
        
        else:
            return (f"Please choose one of the following options:\n"
                    f"• **'name'** - to change your name\n"
                    f"• **'phone'** - to change your phone number\n"
                    f"• **'email'** - to change your email address\n"
                    f"• **'purpose'** - to change the appointment purpose\n"
                    f"• **'date'** - to change the appointment date\n"
                    f"• **'all'** - to start over with all details")

    return "I'm not sure how to proceed. Could you clarify your intent?"

def handle_conversation(query, session_id="default"):
    if vector_store is None:
        return "Please upload a document first before asking questions about it."
    
    relevant_docs = vector_store.similarity_search(query, k=3)
    if not relevant_docs:
        return "Sorry, I couldn't find relevant information in the uploaded document."

    try:
        response = chain_with_history.invoke(
            {
                "input": query,
                "context": "\n".join([doc.page_content for doc in relevant_docs]),
            },
            config={"configurable": {"session_id": session_id}},
        )
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"An error occurred during processing: {str(e)}"

def add_user_message(message,history):
    """Add user message to history immediately"""
    history.append([message,None])
    return history, ""

def get_bot_response(history):
    """Get bot response for the last user message"""
    global COLLECTING_INFO_STATE, APPOINTMENT_STATE, USER_INFO, NEXT_ACTION

    if not history or history[-1][1] is not None:
        return history
    
    message = history[-1][0] ## last user message 

    if COLLECTING_INFO_STATE:
        bot_response = handle_information_collection(message)

    elif APPOINTMENT_STATE:
        bot_response = handle_appointment_booking(message)

    elif is_appointment_request(message):
        bot_response = handle_appointment_booking(message)

    elif is_call_request(message):
        COLLECTING_INFO_STATE = "NAME"
        NEXT_ACTION = "CALLBACK"
        USER_INFO = {}
        bot_response = "Sure! I'd be happy to arrange a callback. May I know your name, please?"

    else:
        bot_response = handle_conversation(message)

    history[-1][1] = bot_response
    return history


def reset_conversation():
    """Reset all conversation states"""
    global COLLECTING_INFO_STATE, APPOINTMENT_STATE, USER_INFO, APPOINTMENT_INFO, NEXT_ACTION, chat_history
    COLLECTING_INFO_STATE = None
    APPOINTMENT_STATE = None
    USER_INFO = {}
    APPOINTMENT_INFO = {}
    NEXT_ACTION = None
    if chat_history:
        chat_history.clear()
    return [], "Conversation reset! You can start fresh."

# Initialize the LLM when the module loads
initialize_llm()

# Create Gradio interface
with gr.Blocks(title="Document AI Chatbot", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🤖 Document AI Chatbot")
    gr.Markdown("Upload a document and chat with an AI assistant that can answer questions, book appointments, and arrange callbacks!")
    
    # Features section moved to the top
    gr.Markdown("""
    ### 🌟 Features:
    - **Document Q&A**: Upload documents and ask questions about their content
    - **Appointment Booking**: Say "book appointment" to schedule meetings with verification
    - **Callback Requests**: Say "call me" to arrange phone callbacks
    - **Smart Context**: Maintains conversation history for natural interactions
    
    ### 📝 Supported File Types:
    - PDF (.pdf) • Text (.txt) • CSV (.csv) • Markdown (.md) • HTML (.html, .htm)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Document Upload")
            file_upload = gr.File(
                label="Upload Document",
                file_types=[".txt", ".pdf", ".csv", ".md", ".html", ".htm"],
                type="filepath"
            )
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                max_lines=3
            )
            upload_message = gr.Textbox(
                label="System Message",
                interactive=False,
                visible=False
            )
            
            gr.Markdown("### 🔄 Actions")
            reset_btn = gr.Button("Reset Conversation", variant="secondary")
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(
                height=600, 
                label="Conversation",
                show_label=True,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Type your message",
                    placeholder="Ask about the document, book an appointment, or request a callback...",
                    container=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Event handlers
    file_upload.upload(
        fn=upload_document,
        inputs=[file_upload],
        outputs=[upload_status, upload_message]
    )
    
    msg_box.submit(
        fn=add_user_message,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box]  # Updated to clear input box
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    send_btn.click(
        fn=add_user_message,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box]  # Updated to clear input box
    ).then(
        fn=get_bot_response,
        inputs=[chatbot],
        outputs=[chatbot]
    )
    
    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[chatbot, upload_message]
    )

if __name__ == "__main__":
    app.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )