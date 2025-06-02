from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,TextLoader,CSVLoader,UnstructuredMarkdownLoader,BSHTMLLoader
from langchain.chains.question_answering.chain import load_qa_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
load_dotenv()
import google.generativeai as genai
import re
from langchain_core.chat_history import InMemoryChatMessageHistory
from datetime import datetime, timedelta
import pytz  # You might need to install this: pip install pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Set your local timezone (adjust as needed)
local_tz = pytz.timezone('Asia/Kathmandu')  # Since you're in Kathmandu

DOCUMENT_PATH = "/home/lucifer/Documents/Agent/docs/5. Automated Personal Writing Pattern Replicator.pdf"

# Load and process documents (same as your existing code)
print(f"Loading document from {DOCUMENT_PATH}")
if DOCUMENT_PATH.endswith('.txt'):
    loader = TextLoader(DOCUMENT_PATH)
elif DOCUMENT_PATH.endswith('.pdf'):
    loader = PyPDFLoader(DOCUMENT_PATH)
elif DOCUMENT_PATH.endswith('.csv'):
    loader = CSVLoader(DOCUMENT_PATH)
elif DOCUMENT_PATH.endswith('.md'):
    loader = UnstructuredMarkdownLoader(DOCUMENT_PATH)
elif DOCUMENT_PATH.endswith(('.html','.htm')):
    loader = BSHTMLLoader(DOCUMENT_PATH)
else:
    raise ValueError("Unsupported document format. Please use .txt, .pdf, or .csv files.")

documents = loader.load()
if not documents:
    print("Could not load any documents. Please check the file path and format.")
    exit()

print(f"Loaded {len(documents)} documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts = text_splitter.split_documents(documents)
if not texts:
    print("Could not split documents into text chunks.")
    exit()

print(f"Split into {len(texts)} text chunks.")

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GEMINI_API_KEY)
    vector_store = FAISS.from_documents(texts,embeddings)
    print("FAISS vector store created successfully.")
except Exception as e:
    print(f"Error creating embeddings or vector store: {e}")
    exit()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=GEMINI_API_KEY)
print("LLM initialized successfully.")

chat_history = InMemoryChatMessageHistory()

# prompt_template = """
# Use context and chat history to answer:
# Context: {context}
# Chat History: {chat_history}
# Question: {input}
# Answer:"""

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
                        input_variables=["context","chat_history","input"])

qa_chain = PROMPT | llm
print("QA chain loaded successfully.")

chain_with_history = RunnableWithMessageHistory(
    qa_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Global state variables
COLLECTING_INFO_STATE = None
APPOINTMENT_STATE = None
USER_INFO = {}
APPOINTMENT_INFO = {}
NEXT_ACTION = None  # NEW: Track what to do after collecting info

# Keywords for different intents
CALL_REQUEST_KEYWORDS = ["call me","callback","contact me","reach out","phone me"]
APPOINTMENT_KEYWORDS = ["book appointment", "schedule appointment", "make appointment", 
                       "book meeting", "schedule meeting", "appointment", "meeting"]

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
        
        # Check if the date is not in the past (optional validation)
        extracted_datetime = datetime.strptime(extracted_date, '%Y-%m-%d')
        today = datetime.now(local_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        
        if extracted_datetime.date() < today.date():
            return None, "The date you specified appears to be in the past. Please provide a future date."
        
        return extracted_date, None
        
    except ValueError:
        return None, "I couldn't understand that date format. Please try again."
    except Exception as e:
        print(f"Error extracting date: {e}")
        return None, "There was an error processing the date. Please try again."

def send_appointment_email(user_info, appointment_info):
    """Send appointment confirmation email"""
    try:
        print(f"Sender email: {SENDER_EMAIL}\n")
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
        status_code,response = server.ehlo()
        print(f"[*] Echoing the server: {status_code} {response}")
        status_code,response = server.starttls()
        print(f"[*] Starting TLS connection: {status_code} {response}")

        status_code,response = server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        print(f"[*] Logging In: {status_code} {response}")

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
        phone_regex = r"^\+?[\d\s\-\(\)]{7,}$"
        if re.fullmatch(phone_regex, query.strip()) and sum(c.isdigit() for c in query) >= 9:
            USER_INFO["phone"] = query.strip()
            COLLECTING_INFO_STATE = "EMAIL"
            return "Great! And finally, what is your email address?"
        else:
            return "Please provide a valid phone number."
    
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
                       "We will contact you shortly. What else can I help you with?")
            else:
                return (f"Thank you, {USER_INFO.get('name', 'User')}! We have your details:\n"
                       f"Phone: {USER_INFO.get('phone')}\n"
                       f"Email: {USER_INFO.get('email')}\n"
                       "Your information has been saved. What else can I help you with?")
        else:
            return "That doesn't look like a valid email address. Please enter a valid email address (e.g., user@example.com)."
    
    return "I'm not sure. Could you please clarify?"


def handle_appointment_booking(query):
    global APPOINTMENT_STATE, APPOINTMENT_INFO, USER_INFO, COLLECTING_INFO_STATE,NEXT_ACTION

    # If we haven't started the appointment process yet
    if not APPOINTMENT_STATE:
        if USER_INFO and USER_INFO.get('name') and USER_INFO.get('phone') and USER_INFO.get('email'):  # Check if user details already exist
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

    # Handle user's choice about info
    if APPOINTMENT_STATE == "AWAITING_INFO_CHOICE":
        lower_query = query.lower()
        if "use previous" in lower_query:
            APPOINTMENT_STATE = "PURPOSE"
            return "Great! What is the purpose of your appointment?"
        elif "new" in lower_query:
            USER_INFO.clear()  # Reset old info
            COLLECTING_INFO_STATE = "NAME"
            APPOINTMENT_STATE = None # Reset appointment state to re-enter
            NEXT_ACTION = "APPOINTMENT"
            return "Alright, let's start with your new details. What is your full name?"
        else:
            return "Please reply with either 'use previous' or 'new details'."

    # Proceed with purpose collection
    if APPOINTMENT_STATE == "PURPOSE":
        APPOINTMENT_INFO["purpose"] = query
        APPOINTMENT_STATE = "DATE"
        return "Got it. What date would you prefer for the appointment?"

    elif APPOINTMENT_STATE == "DATE":
        date_str, error = extract_date_llm(query)
        if error:
            return error
        APPOINTMENT_INFO["date"] = date_str
        APPOINTMENT_STATE = None  # Reset
        if send_appointment_email(USER_INFO, APPOINTMENT_INFO):
            return (f"Your appointment for {APPOINTMENT_INFO['purpose']} on {APPOINTMENT_INFO['date']} "
                    f"has been booked successfully! A confirmation email has been sent to {USER_INFO['email']}.")
        else:
            return "Your appointment was booked, but there was an error sending the confirmation email."

    return "I'm not sure how to proceed. Could you clarify your intent?"


def handle_conversation(query, session_id="default"):
    relevant_docs = vector_store.similarity_search(query, k=3)
    if not relevant_docs:
        return "Sorry, I couldn't find relevant information."

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
        print(f"Error: {e}")
        return "An error occurred during processing."

def main():
    global COLLECTING_INFO_STATE, APPOINTMENT_STATE, USER_INFO, NEXT_ACTION
    session_id = "default"
    
    print("Chatbot is ready! Type 'exit' to quit.")
    print("You can ask questions about the document, request a callback, or book an appointment.")
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        bot_response = ""

        # Handle ongoing information collection (highest priority)
        if COLLECTING_INFO_STATE:
            bot_response = handle_information_collection(user_query)
        
        # Handle ongoing appointment booking (second highest priority)
        elif APPOINTMENT_STATE:
            bot_response = handle_appointment_booking(user_query)
        
        # Handle new appointment request
        elif is_appointment_request(user_query):
            # The logic to decide whether to ask for new/previous info is now primarily
            # handled within handle_appointment_booking, by checking USER_INFO there.
            bot_response = handle_appointment_booking(user_query)
        
        # Handle new call request
        elif is_call_request(user_query):
            COLLECTING_INFO_STATE = "NAME"
            NEXT_ACTION = "CALLBACK"  # Remember we're collecting info for callback
            USER_INFO = {}  # Reset user info to ensure fresh collection for callback
            bot_response = "Sure! I'd be happy to arrange a callback. May I know your name, please?"
        
        # Handle regular document queries
        else:
            bot_response = handle_conversation(user_query, session_id=session_id)
        
        print(f"Bot: {bot_response}")
        print("-" * 100)

if __name__ == "__main__":
    main()
