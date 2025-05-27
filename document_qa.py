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

# Create the QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")
print("QA chain loaded successfully.")



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
        if query.replace("+","").replace(" ", "").isdigit() and len(query.replace(" ","")) == 10:
            USER_INFO["phone"] = query
            COLLECTING_INFO_STATE = "EMAIL"
            response = "Great. And finally, what is your email address?"
        else:
            response = "Please provide a valid phone number."
    elif COLLECTING_INFO_STATE == "EMAIL":
        if "@" in query and "." in query.split("@")[-1]:
            USER_INFO["email"] = query
            response = (f"Thank you, {USER_INFO.get('name', 'User')}! I got your details:\n "
                        f"Phone: {USER_INFO.get('phone', 'Not provided')}\n"
                        f"Email: {USER_INFO.get('email', 'Not provided')}\n")
            print(f"Collected user information: {USER_INFO}")
            COLLECTING_INFO_STATE = None
            USER_INFO = {}
        else:
            response = "Please provide a valid email address."
    else:
        response = "I'm not sure. Could you please clarify ?"
        COLLECTING_INFO_STATE = None

    return response



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





## Example 
# if __name__ == "__main__":
#     while True:
#         user_query = input("\nEnter your question (or type 'exit' to quit): ")
#         if user_query.lower() == 'exit':
#             print("Exiting the program.")
#             break
#         answer = ask_question(user_query)
#         if answer is None:
#             print("No answer found for your question.")
#         else:
#             print(f"Answer: {answer}")
#         print("You can ask another question or type 'exit' to quit.")
#         print("-" * 100)



if __name__ == "__main__":
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Exiting the program.")
            break

        bot_response = ""

        if COLLECTING_INFO_STATE:
            bot_response = handle_information_collection(user_query)
        elif is_call_request(user_query):
            COLLECTING_INFO_STATE = "NAME"
            USER_INFO = {}
            bot_response = "Sure! May I know your name, please?"
        else:
            bot_response = ask_question(user_query)
            if not bot_response:
                bot_response = "I'm sorry, I couldn't find an answer to your question."
        print(f"Bot: {bot_response}")
        print("-" * 100)