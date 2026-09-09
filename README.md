# 🏗️ Construction Project AI Assistant

## 1. Project Overview

The **Construction Project AI Assistant** is a Generative AI application that helps users find information from construction project documents using natural-language questions.

Construction projects usually contain many documents such as drawings, specifications, RFIs, and other project PDFs. Finding specific information manually can be time-consuming.

This application allows the user to **upload the project documents once and ask questions directly**.

For example:

> "Give me the reinforcement details for F1 footing."

The system searches the project documents, finds the relevant information, and generates an answer with the **source file and page number**. For drawing-related questions, it can also display the relevant drawing page.

---

## 2. Why Did We Create This?

The main problem we are solving is **finding information quickly inside large construction document collections**.

Without this application:

```text
Open many PDFs
     ↓
Search manually
     ↓
Find the correct page
     ↓
Understand the information
     ↓
Give the answer
```

With this application:

```text
Ask a question
     ↓
AI searches project documents
     ↓
Relevant information is retrieved
     ↓
AI generates the answer
     ↓
Source + page are shown
```

The goal is to make project-document search **faster, easier, and more accessible**, even for users who are not familiar with the project's document structure.

---

## 3. How Does It Work?

The application uses **RAG (Retrieval-Augmented Generation)**.

In simple terms, RAG means:

> **First find the relevant information from the user's documents, then give that information to the AI to generate the answer.**

### Workflow

```text
Project ZIP
    ↓
Extract PDF Documents
    ↓
Read PDF Pages
    ↓
Split Text into Chunks
    ↓
Create Embeddings
    ↓
Store in FAISS
    ↓
User Asks Question
    ↓
Retrieve Relevant Information
    ↓
AI Generates Answer
    ↓
Answer + Sources + Drawing Pages
```

This approach helps the AI answer based on the **actual project documents** rather than relying only on its general knowledge.

---

## 4. Key Features

### 📁 Project Upload

Upload the project documents as a ZIP file.

### 🔍 Intelligent Search

Search project information using natural-language questions.

### 🧠 RAG

Retrieves relevant project information before generating the answer.

### 🔎 Hybrid Retrieval

Uses semantic and keyword-based search to improve document retrieval.

### 📄 Source References

Shows the document name and page number used for the answer.

### 🏗️ Drawing Search

Can identify relevant drawing pages for drawing-related questions.

### 👁️ Visual Understanding

Relevant drawing pages can be rendered and provided to the AI for visual questions.

### 💾 Project Cache

Previously processed projects can be loaded without processing the same project again.

### ⚡ Answer Cache

Previously answered questions can be reused to reduce unnecessary AI calls.

### 💬 Streamlit Interface

Provides a simple interface for uploading projects, asking questions, viewing answers, and checking sources.

---

## 5. Technologies Used

| Technology          | Purpose                   |
| ------------------- | ------------------------- |
| Python              | Application development   |
| Streamlit           | User interface            |
| LangChain           | RAG and retrieval         |
| OpenAI GPT-5.6 Luna | Answer generation         |
| OpenAI Embeddings   | Text embeddings           |
| FAISS               | Vector search             |
| BM25                | Keyword search            |
| PyMuPDF             | PDF reading and rendering |
| JSON                | Answer caching            |

---

## 6. Example Questions

Users can ask questions such as:

```text
Give me the reinforcement details for F1 footing.

Which page contains the beam reinforcement details?

What is the concrete grade mentioned in the specification?

Which drawing contains the foundation plan?

Give me the details available for column C5.
```

The user does not need to know the file name or page number beforehand.

---

## 7. Running the Application

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add OpenAI API Key

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key
```

### Start the application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 8. Important Limitation

This application is a **document search and AI assistance tool**.

It should not replace a qualified engineer's professional judgment.

For structural design, safety, construction approval, calculations, or code compliance, the original project documents and qualified professionals should always be consulted.

---

## 9. Future Scope

Possible future improvements include:

* Better OCR for scanned drawings
* Improved drawing and table understanding
* Drawing revision comparison
* RFI analysis
* Document comparison
* Multiple project management
* User authentication
* Cloud deployment
* Advanced guardrails
* RAG evaluation and monitoring
* AI agents for construction workflows

---

## 10. Project Goal

> **To build an AI assistant that allows users to quickly find, understand, and verify information from large construction project documents using natural-language questions.**
