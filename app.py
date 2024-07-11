from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import mysql.connector

app = Flask(__name__)

# Load the CSV file
df = pd.read_csv('employee_queries_dataset_large.csv')

# Clean the 'Prompt' column to remove NaNs and non-string values
df['Prompt'] = df['Prompt'].fillna('').astype(str)

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the 'Prompt' column
df['embeddings'] = df['Prompt'].apply(lambda x: model.encode(x))

# Convert embeddings to a format suitable for FAISS
embeddings = np.stack(df['embeddings'].values.tolist())

# Initialize FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Function to encode a query using Sentence Transformer
def encode_query(query, model):
    return model.encode(query)

def semantic_search_top_query(query, model, index, df, threshold=0.8):
    query_embedding = encode_query(query, model)
    D, I = index.search(np.array([query_embedding]), k=1)
    distance = D[0][0]
    retrieved_query = df.iloc[I[0][0]]['Query']

    # Check if the distance is below the threshold
    if distance <= threshold:
        return retrieved_query
    else:
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

def generate_response_with_bart(results):
    input_text = " ".join([str(row) for row in results])
    input_ids = BartTokenizer.from_pretrained('facebook/bart-large').encode(input_text, return_tensors='pt')

    # Load BART model and tokenizer
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    with torch.no_grad():
        output_ids = bart_model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)

    response = BartTokenizer.from_pretrained('facebook/bart-large').decode(output_ids[0], skip_special_tokens=True)
    return response

@app.route('/process_query', methods=['POST'])
def process_query():
    user_query = request.json.get('query')
    retrieved_query = semantic_search_top_query(user_query, model, index, df)
    if retrieved_query:
        results = execute_sql_query(retrieved_query)
        if results:
            response = generate_response_with_bart(results)
            return jsonify({"response": response})
        else:
            return jsonify({"error": "No results found for the executed query."})
    else:
        return jsonify({"error": "No relevant query found."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
