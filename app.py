from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import mysql.connector
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize global variables
df = None
model = None
index_main = None
d = None

def initialize():
    global df, model, index_main, d

    # Load the CSV file
    df = pd.read_csv('employee_queries_dataset_large.csv')

    # Clean the 'Prompt' column to remove NaNs and non-string values
    df['Prompt'] = df['Prompt'].fillna('').astype(str)

    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the 'Prompt' column
    df['embeddings'] = df['Prompt'].apply(lambda x: model.encode(x) if isinstance(x, str) else None)

    # Filter out rows where embeddings could not be generated
    df = df.dropna(subset=['embeddings'])

    # Convert embeddings to a format suitable for FAISS
    embeddings = np.stack(df['embeddings'].values.tolist())

    # Initialize FAISS index
    d = embeddings.shape[1]
    index_main = faiss.IndexFlatL2(d)
    index_main.add(embeddings)

# Initialize on startup
initialize()

# Function to encode a query using Sentence Transformer
def encode_query(query, model):
    if query and isinstance(query, str):
        return model.encode(query)
    return None

def semantic_search_top_query(query, model, index, df, threshold=0.8):
    query_embedding = encode_query(query, model)
    if query_embedding is None:
        return None
    
    D, I = index.search(np.array([query_embedding]), k=1)
    
    if I is None or len(I) == 0 or len(I[0]) == 0:
        return None
    
    distance = D[0][0]
    index_position = I[0][0]
    
    if 0 <= index_position < len(df):
        retrieved_query = df.iloc[index_position]['Query']
        
        # Check if the distance is below the threshold
        if distance <= threshold:
            return retrieved_query
    
    return None

# Placeholder function to execute an SQL query
def execute_sql_query(query):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',  # Adjust host if necessary
            user='root',  # Replace with your MySQL username
            password='1234',  # Replace with your MySQL password
            database='task1'  # Database name  
        )
        cursor = connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        connection.close()
        print("successfully connected to the mysql")
        return results
    except Exception as e:
        print(f"An error occurred while executing the SQL query: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    logging.info("Received request for process_query")
    user_prompt = request.json.get('prompt')
    if not user_prompt:
        logging.error("No prompt provided")
        return jsonify({"error": "Prompt is required in the request."}), 400
    
    retrieved_query = semantic_search_top_query(user_prompt, model, index_main, df)
    
    if retrieved_query:
        results = execute_sql_query(retrieved_query)
        
        if results:
            return jsonify({"response": results})
        else:
            logging.error("No results found for the executed query")
            return jsonify({"error": "No results found for the executed query."}), 404
    else:
        logging.error("No relevant query found")
        return jsonify({"error": "No relevant query found."}), 404

@app.route('/add_client_dataset', methods=['POST'])
def add_client_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file:
        try:
            client_df = pd.read_csv(file)

            # Validate columns
            if 'Prompt' not in client_df.columns or 'Query' not in client_df.columns:
                return jsonify({"error": "CSV must contain 'Prompt' and 'Query' columns."}), 400

            # Clean the 'Prompt' column to remove NaNs and non-string values
            client_df['Prompt'] = client_df['Prompt'].fillna('').astype(str)

            # Generate embeddings for the 'Prompt' column
            client_df['embeddings'] = client_df['Prompt'].apply(lambda x: model.encode(x) if isinstance(x, str) else None)

            # Filter out rows where embeddings could not be generated
            client_df = client_df.dropna(subset=['embeddings'])

            # Convert embeddings to a format suitable for FAISS
            client_embeddings = np.stack(client_df['embeddings'].values.tolist())

            # Add new data to the main dataframe
            global df, index_main
            df = pd.concat([df, client_df], ignore_index=True)

            # Add new embeddings to the FAISS index
            index_main.add(client_embeddings)

            return jsonify({"message": "Client dataset added successfully."}), 200
        except Exception as e:
            return jsonify({"error": f"An error occurred while processing the file: {e}"}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, threaded=True)
