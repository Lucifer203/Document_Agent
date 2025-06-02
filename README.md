
# 🤖 Document AI Chatbot

This is a versatile AI chatbot built with Gradio and Langchain, powered by Google's Gemini models. It allows users to upload various document types and then interact with the content through natural language. Beyond document Q&A, the chatbot is also equipped to handle specific intents like **appointment booking** and **callback requests**, providing a more integrated and helpful user experience.

---

## ✨ Features

* **Intelligent Document Q&A:** Upload PDF, TXT, CSV, Markdown, or HTML files and ask questions about their content. The chatbot uses **FAISS** for efficient similarity search to find relevant information.
* **Context-Aware Conversations:** Leverages **Langchain's `RunnableWithMessageHistory`** to maintain conversational context, allowing for natural follow-up questions and more coherent interactions.
* **Automated Appointment Booking:** Users can initiate appointment scheduling. The chatbot guides them through collecting necessary details (name, phone, email, purpose, date) and verifies the information before confirming. Confirmation emails are sent to the user.
* **Seamless Callback Requests:** Users can ask for a callback, prompting the chatbot to collect their contact information.
* **User-Friendly Interface:** Built with **Gradio** for an intuitive web-based chat interface.

---

## 🛠️ Technologies Used

* **Python**
* **Google Gemini API:** For powerful language understanding and generation (`gemini-1.5-flash` model for chat, `embedding-001` for embeddings).
* **Langchain:** Framework for building LLM-powered applications.
    * `langchain-google-genai`
    * `langchain-core`
    * `langchain.vectorstores.FAISS`
    * `langchain.text_splitter.RecursiveCharacterTextSplitter`
    * `langchain.document_loaders` (PyPDFLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader, BSHTMLLoader)
* **Gradio:** For creating the interactive web UI.
* **python-dotenv:** For managing environment variables.
* **pytz:** For timezone handling.
* **smtplib & email.mime:** For sending email confirmations.

---

## 🚀 Setup and Installation

Follow these steps to get your Document AI Chatbot up and running locally:

### 1. Clone the Repository

```bash
git clone git@github.com:Lucifer203/Document_Agent.git
cd Document_Agent
````

### 2\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

  * **Linux / macOS:**

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

  * **Windows (Command Prompt):**

    ```bash
    python -m venv myenv
    myenv\Scripts\activate.bat
    ```

  * **Windows (PowerShell):**

    ```powershell
    python -m venv myenv
    myenv\Scripts\Activate.ps1
    ```

### 3\. Install Dependencies

Once your virtual environment is active, install the required Python packages:

```bash
pip install -r requirements.txt
```
### 4\. Configure Environment Variables

Create a file named `.env` in the root directory of your project (the same directory as `app.py`) and add the following:

```
GEMINI_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
SENDER_EMAIL="YOUR_SENDER_EMAIL@gmail.com"
EMAIL_PASSWORD="YOUR_APP_PASSWORD"
```

**Important Notes:**

  * **`GEMINI_API_KEY`**: Obtain this from the [Google AI Studio](https://ai.google.dev/).
  * **`SENDER_EMAIL`**: This should be a Gmail address you control.
  * **`EMAIL_PASSWORD`**: This **must be an App Password**, not your regular Gmail password.
      * To generate an App Password:
        1.  Go to your Google Account.
        2.  Navigate to Security.
        3.  Under "How you sign in to Google," select **2-Step Verification** and ensure it's **ON**.
        4.  Below "2-Step Verification," you'll find **App passwords**. Click on it.
        5.  Follow the instructions to generate a new app password. You might need to select "Mail" for the app and "Other (Custom name)" for the device. Copy the generated password (it's usually a 16-character string).

### 5\. Run the Application

With your virtual environment active and `.env` configured, run the Gradio application:

```bash
python app.py
```

The application will start, and you'll see a local URL (e.g., `http://127.0.0.1:7860`) and potentially a public Gradio Share URL in your terminal. Open these URLs in your web browser to access the chatbot interface.

-----

## 🤖 How to Use

1.  **Upload a Document:** Use the "Upload Document" section on the left to upload your `.pdf`, `.txt`, `.csv`, `.md`, or `.html` files. The "Upload Status" will indicate success or failure.
2.  **Ask Questions:** Once a document is loaded, type your questions into the chat input box and press "Send" or Enter.
3.  **Book an Appointment:** Type phrases like "book appointment" or "schedule meeting." The bot will guide you through providing your name, phone, email, purpose, and preferred date. It will summarize the details for confirmation and send an email upon successful booking.
4.  **Request a Callback:** Type phrases like "call me" or "contact me." The bot will ask for your name, phone, and email to arrange a callback.
5.  **Reset Conversation:** Click the "Reset Conversation" button to clear the chat history and internal states, allowing you to start a fresh interaction.

-----

## 📝 Document Loading Notes

The chatbot supports a variety of document types by leveraging different Langchain loaders:

  * `.txt`: `TextLoader`
  * `.pdf`: `PyPDFLoader`
  * `.csv`: `CSVLoader`
  * `.md`: `UnstructuredMarkdownLoader`
  * `.html`, `.htm`: `BSHTMLLoader`

Documents are chunked using `RecursiveCharacterTextSplitter` and stored in a **FAISS** vector store for efficient retrieval during conversations.

-----
