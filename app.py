from flask import Flask, request, jsonify, render_template_string
from pdf_processing import extract_text_from_pdf, clean_text, preprocess_text
from qa_engine import find_best_answer
from sentence_transformers import SentenceTransformer
import os

# Initialize Flask app and global variables
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables
pdf_chunks, chunk_embeddings = None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    global pdf_chunks, chunk_embeddings
    answer = None

    if request.method == 'POST':
        # Handle PDF upload
        if 'pdf' in request.files:
            file = request.files['pdf']
            if file.filename.endswith('.pdf'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                try:
                    pdf_text = extract_text_from_pdf(file_path)
                    pdf_text = clean_text(pdf_text)
                    pdf_chunks = preprocess_text(pdf_text)
                    chunk_embeddings = model.encode(pdf_chunks)
                    answer = "PDF uploaded and processed successfully! You can now ask questions."
                except Exception as e:
                    answer = f"Error processing PDF: {e}"
                    pdf_chunks, chunk_embeddings = None, None  # Reset on error
            else:
                answer = "Invalid file format. Please upload a PDF file."

        # Handle query input
        elif 'query' in request.form:
            query = request.form.get('query', '')
            if pdf_chunks is not None and len(pdf_chunks) > 0 and chunk_embeddings is not None and len(chunk_embeddings) > 0:
                try:
                    answer = find_best_answer(query, pdf_chunks, chunk_embeddings, model)
                except Exception as e:
                    answer = f"Error finding answer: {e}"
            else:
                answer = "No PDF uploaded. Please upload a PDF first."

    # Render the form and results
    return render_template_string('''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF Chatbot</title>
        <!-- Bootstrap CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f0f2f5;
                font-family: Arial, sans-serif;
            }
            .main-container {
                margin-top: 50px;
                max-width: 800px;
                background: #ffffff;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                font-weight: bold;
                color: #343a40;
            }
            .btn-primary {
                background-color: #007bff;
                border: none;
                font-weight: bold;
                width: 100%;
                padding: 10px 15px;
            }
            .btn-primary:hover {
                background-color: #0056b3;
            }
            .answer-box {
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-left: 5px solid #007bff;
                border-radius: 5px;
            }
            .custom-divider {
                border-top: 2px solid #007bff;
                margin: 30px 0;
            }
        </style>
    </head>
    <body>
        <div class="container main-container">
            <h1 class="text-center">PDF Chatbot</h1>
            <p class="text-center text-muted">Upload a PDF and ask questions about its content.</p>
            
            <!-- PDF Upload Section -->
            <form method="post" enctype="multipart/form-data" class="mb-4">
                <div class="mb-3">
                    <label for="pdf" class="form-label">Upload Your PDF</label>
                    <input type="file" id="pdf" name="pdf" accept="application/pdf" class="form-control">
                </div>
                <button type="submit" class="btn btn-primary">Upload PDF</button>
            </form>

            <div class="custom-divider"></div>

            <!-- Question Section -->
            <form method="post" class="mb-4">
                <div class="mb-3">
                    <label for="query" class="form-label">Ask a Question</label>
                    <input type="text" id="query" name="query" placeholder="What would you like to know?" class="form-control" required>
                </div>
                <button type="submit" class="btn btn-primary">Search</button>
            </form>

            <!-- Display Answer -->
            {% if answer %}
                <div class="answer-box">
                    <h4>Answer:</h4>
                    <p>{{ answer }}</p>
                </div>
            {% endif %}
        </div>

        <!-- Bootstrap JS (Optional) -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
''', answer=answer)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
