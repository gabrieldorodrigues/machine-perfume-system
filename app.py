from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load vectorizer and dataset
vectorizer = joblib.load('model/vectorizer.pkl')
df = pd.read_csv('data/dataset_tratado.csv')
df_vectors = pd.read_csv('data/notes_vectors.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    perfume_name = request.form['perfume_name']

    # Find the perfume in the dataset
    if perfume_name not in df['name'].values:
        return jsonify({"error": "Perfume not found"}), 404

    # Get the vector for the selected perfume
    perfume_index = df[df['name'] == perfume_name].index[0]
    perfume_vector = df_vectors.iloc[perfume_index].values.reshape(1, -1)

    # Calculate cosine similarities
    similarities = cosine_similarity(perfume_vector, df_vectors).flatten()

    # Get the top 10 most similar perfumes (excluding itself)
    similar_indices = similarities.argsort()[::-1][1:10]
    similar_scores = similarities[similar_indices]
    similar_perfumes = df.iloc[similar_indices][['brand', 'name', 'notes']]

    # Prepare response
    recommendations = similar_perfumes.to_dict(orient='records')
    for i in range(len(recommendations)):
        recommendations[i]['similarity'] = round(similar_scores[i] * 100, 2)

    selected_perfume = df.iloc[perfume_index][['brand', 'name', 'notes']].to_dict()
    
    return jsonify({
        'selected_perfume': selected_perfume,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
