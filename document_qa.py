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

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables")


genai.configure(api_key=GEMINI_API_KEY) 

# DOCUMENT_PATH = "./docs/testing_data.txt"
# DOCUMENT_PATH = "/home/lucifer/Documents/Agent/docs/CoverLetter.pdf"
DOCUMENT_PATH = "/home/lucifer/Documents/Agent/docs/trial.html"



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

## Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts = text_splitter.split_documents(documents)
if not texts:
    print("Could not split documents into text chunks.")
    exit()

print(f"Split into {len(texts)} text chunks.")

# Create embeddings
print("Creating embeddings and FAISS vector store")
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GEMINI_API_KEY)
    vector_store = FAISS.from_documents(texts,embeddings)
    print("FAISS vector store created successfully.")
except Exception as e:
    print(f"Error creating embeddngs or vector store: {e}")
    print("Please check your API key and network connection.")
    exit()

## Initializing the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=GEMINI_API_KEY)
print("LLM initialized successfully.")

## initializing memory
# memory = ConversationBufferMemory(
#     return_messages=True,
#     input_key="input",
#     output_key="output"
# )

chat_history = InMemoryChatMessageHistory()

prompt_template = """
Use context and chat history to answer:
Context: {context}
Chat History: {chat_history}
Question: {input}
Answer:"""

PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["context","chat_history","input"])

# Create the QA chain
# qa_chain = load_qa_chain(llm, chain_type="stuff")
qa_chain = PROMPT | llm
print("QA chain loaded successfully.")

chain_with_history = RunnableWithMessageHistory(
    qa_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)



COLLECTING_INFO_STATE = None
USER_INFO = {}

CALL_REQUEST_KEYWORDS = ["call me","callback","contact me","reach out","phone me"]

def is_call_request(query):
    query_lower = query.lower()
    for keyword in CALL_REQUEST_KEYWORDS:
        if keyword in query_lower:
            return True
    return False

def handle_information_collection(query):
    global COLLECTING_INFO_STATE,USER_INFO

    response = ""
    if COLLECTING_INFO_STATE == "NAME":
        USER_INFO["name"] = query
        COLLECTING_INFO_STATE = "PHONE"
        response = "Thank you, {name}. What is your phone number?"
    elif COLLECTING_INFO_STATE == "PHONE":
        phone_regex = r"^\+?[\d\s\-\(\)]{7,}$" 
        if re.fullmatch(phone_regex,query.strip()) and sum(c.isdigit() for c in query) >= 9:
            USER_INFO["phone"] = query.strip()
            COLLECTING_INFO_STATE = "EMAIL"
            response = "Great. And finally, what is your email address?"


        else:
            response = "Please provide a valid phone number."
    elif COLLECTING_INFO_STATE == "EMAIL":
        # Regex for a common email pattern
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if re.fullmatch(email_regex, query.strip()):
            USER_INFO["email"] = query.strip()
            response = (f"Thank you, {USER_INFO.get('name', 'User')}! We have your details:\n"
                        f"Phone: {USER_INFO.get('phone')}\n"
                        f"Email: {USER_INFO.get('email')}\n"
                        "Ok I got your information. What else can I help you with?")
            print(f"Collected User Info: {USER_INFO}")
            COLLECTING_INFO_STATE = None # Reset state
            # USER_INFO = {} # We might want to keep user_info if memory is to retain it across turns after collection.
                        # For now, let's clear it as per previous logic, memory will store the conversation.
        else:
            response = "That doesn't look like a valid email address. Please enter a valid email address (e.g., user@example.com)."
  
    else:
        response = "I'm not sure. Could you please clarify ?"
        COLLECTING_INFO_STATE = None

    return response




def handle_conversation(query, session_id="default"):
    relevant_docs = vector_store.similarity_search(query, k=3)
    if not relevant_docs:
        return "Sorry, I couldn't find relevant info."

    try:
        response = chain_with_history.invoke(
            {
                "input": query,
                "context": "\n".join([doc.page_content for doc in relevant_docs]),
                # "chat_history": memory.chat_memory.messages,
            },
            config={"configurable": {"session_id": session_id}},
        )
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during processing."


# Function to answer questions
def ask_question(query):
    print(f"\nSearching answer for: {query}")
    try:
        relevant_docs = vector_store.similarity_search(query, k=3)
        if not relevant_docs:
            print("No relevant documents found.")
            return None
        print(f"Found {len(relevant_docs)} relevant documents.")
        answer = qa_chain.run(input_documents=relevant_docs, question=query)
        return answer
    except Exception as e:
        print(f"Error during question answering: {e}")




# if __name__ == "__main__":
#     session_id = "default"
#     while True:
#         user_query = input("\nYou: ")
#         if user_query.lower() == 'exit':
#             print("Exiting the program.")
#             break

#         bot_response = ""

#         if COLLECTING_INFO_STATE:
#             bot_response = handle_information_collection(user_query)
#         elif is_call_request(user_query):
#             COLLECTING_INFO_STATE = "NAME"
#             USER_INFO = {}
#             bot_response = "Sure! May I know your name, please?"
#         else:
#             bot_response = ask_question(user_query)
#             if not bot_response:
#                 bot_response = "I'm sorry, I couldn't find an answer to your question."
#         print(f"Bot: {bot_response}")
#         print("-" * 100)


if __name__ == "__main__":
    session_id = "default"  # Could be dynamic per user
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Exiting the program.")
            break

        bot_response = ""

        if COLLECTING_INFO_STATE:
            bot_response = handle_information_collection(user_query)
            # Store user info in memory if completed
            if COLLECTING_INFO_STATE is None and USER_INFO:
                user_info_message = f"My name is {USER_INFO['name']}, phone is {USER_INFO['phone']}, and email is {USER_INFO['email']}."
                chat_history.add_user_message(user_info_message)
                # memory.chat_memory.add_user_message(
                #     f"My name is {USER_INFO['name']}, phone is {USER_INFO['phone']}, and email is {USER_INFO['email']}."
                # )
        elif is_call_request(user_query):
            COLLECTING_INFO_STATE = "NAME"
            USER_INFO = {}
            bot_response = "Sure! May I know your name, please?"
        else:
            bot_response = handle_conversation(user_query, session_id=session_id)
        print(f"Bot: {bot_response}")
        print("-" * 100)
        # print(memory.input_key)
