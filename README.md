# AI Bot: PDF-Based Question-Answering Chatbot

## Overview
AI Bot is a web application designed to interactively extract and answer questions based on the content of PDF documents. Users can upload a PDF, and the bot intelligently retrieves answers using state-of-the-art natural language processing (NLP) techniques.

---

## Features
- **PDF Upload**: Upload a text-based PDF document for processing.
- **Intelligent Q&A**: Ask questions related to the uploaded document and get context-aware responses.
- **Responsive Design**: User-friendly, responsive interface built with Bootstrap.
- **Real-Time Processing**: Quick and efficient answer retrieval with minimal latency.

---

## Technology Stack
### Backend:
- **Flask**: Web framework for the backend server.
- **pdfplumber**: Extract text content from PDF documents.
- **Sentence Transformers**: Generate embeddings for semantic similarity.
- **NumPy**: Perform mathematical computations, such as cosine similarity.

### Frontend:
- **HTML & CSS**: Structure and style the web interface.
- **Bootstrap**: Enhance design and ensure responsiveness.

---

## System Workflow
1. **PDF Upload**:
   - The user uploads a PDF document via the web interface.
   - Text is extracted, cleaned, and divided into smaller chunks for easier processing.

2. **Embedding Generation**:
   - Each chunk is transformed into a vector representation using Sentence Transformers.
   - These embeddings are stored for quick similarity calculations.

3. **Query Processing**:
   - When the user submits a query, it is converted into an embedding.
   - Cosine similarity scores are calculated between the query embedding and the document embeddings.

4. **Answer Retrieval**:
   - The chunk with the highest similarity score is selected.
   - The most relevant sentence from the chunk is displayed as the answer.

---

## Installation and Setup
Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone <repository-url>
cd aibot-project
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Application
```bash
python app.py
```
### 4. Access the App
   - http://127.0.0.1:5000, or
   - http://your_local_address:500

## Usage

### 1. Upload a PDF:
  - Use the "Upload PDF" form on the homepage to upload your document.
  - Once uploaded, the bot processes the document and prepares it for answering questions.
### 2. Ask Questions:
  - Type a query related to the document in the text input box.
  - The bot will respond with the most relevant answer extracted from the document.

### 3. Setup
```plaintext
aibot-project/
├── app.py               # Main Flask application
├── requirements.txt     # Python package dependencies
├── uploads/             # Directory for storing uploaded PDFs
├── static/              # Directory for static files (CSS, JS, etc.)
├── templates/           # HTML templates
└── README.md            # Project documentation
```

## License

  - This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or further assistance, feel free to reach out:

   - Name: Puneet Kumar Rajan
   - Email: puneetrajan997@gmail.com
