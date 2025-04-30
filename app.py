import pickle
import requests
from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='/static')

# Load movie data and similarity matrix
movies = pickle.load(open('model/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

# Fetch movie poster from TMDB
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Image"

# Recommend movies based on similarity
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append({
            'title': movies.iloc[i[0]].title,
            'poster': fetch_poster(movie_id)
        })
    return recommended_movies

@app.route('/', methods=['GET', 'POST'])
def index():
    movie_list = movies['title'].values
    recommendations = []
    selected_movie = None
    
    if request.method == 'POST':
        selected_movie = request.form['movie']
        recommendations = recommend(selected_movie)
    
    return render_template('index.html', movie_list=movie_list, selected_movie=selected_movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
