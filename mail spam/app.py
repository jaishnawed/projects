from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the trained models and vectorizer
with open("model.pkl", "rb") as f:
    models = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    accuracy = None

    if request.method == "POST":
        email_features = request.form["email_text"]
        model_choice = request.form["model"]

        # Convert email text into numerical form (scaling)
        email_vector = np.array(email_features.split(), dtype=float).reshape(1, -1)
        email_vector = scaler.transform(email_vector)

        # Make prediction
        selected_model = models[model_choice]
        prediction = selected_model.predict(email_vector)[0]
        accuracy = selected_model.score(email_vector, [prediction])

        result_text = "Spam" if prediction == 1 else "Not Spam"
        
        return jsonify({"result": result_text, "accuracy": f"{accuracy * 100:.2f}%"})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
