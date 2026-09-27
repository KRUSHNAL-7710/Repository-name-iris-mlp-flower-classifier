from flask import Flask, request
import numpy as np
import os

app = Flask(__name__)

# Path to trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "iris_mlp_model.npz")

# Load trained model
model = np.load(model_path)

W1 = model["W1"]
b1 = model["b1"]

W2 = model["W2"]
b2 = model["b2"]

W3 = model["W3"]
b3 = model["b3"]

mean = model["mean"]
scale = model["scale"]

classes = ["Setosa", "Versicolor", "Virginica"]


# ReLU activation
def relu(x):
    return np.maximum(0, x)


# Softmax activation
def softmax(x):
    exp_values = np.exp(
        x - np.max(x, axis=1, keepdims=True)
    )
    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )


# Prediction function
def predict_flower(features):

    X = np.array(features, dtype=float)

    # Standardization
    X = (X - mean) / scale

    X = X.reshape(1, -1)

    # Forward propagation
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3

    probabilities = softmax(Z3)

    predicted_class = np.argmax(
        probabilities,
        axis=1
    )[0]

    confidence = (
        probabilities[0][predicted_class] * 100
    )

    return (
        classes[predicted_class],
        confidence
    )


# Home page
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""

    if request.method == "POST":

        try:

            sepal_length = float(
                request.form["sepal_length"]
            )

            sepal_width = float(
                request.form["sepal_width"]
            )

            petal_length = float(
                request.form["petal_length"]
            )

            petal_width = float(
                request.form["petal_width"]
            )

            prediction, confidence = predict_flower(
                [
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                ]
            )

        except Exception:

            prediction = "Invalid Input"
            confidence = ""

    confidence_text = ""

    if confidence != "":
        confidence_text = (
            "Confidence: "
            + str(round(confidence, 2))
            + "%"
        )

    html = '''
    <!DOCTYPE html>

    <html>

    <head>

        <title>Iris Flower Classification</title>

        <style>

            body {
                font-family: Arial;
                background-color: #f4f6f8;
                text-align: center;
                padding: 40px;
            }

            .container {
                background: white;
                width: 450px;
                margin: auto;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            }

            input {
                width: 90%;
                padding: 10px;
                margin: 8px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }

            button {
                padding: 12px 25px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }

            button:hover {
                background-color: #45a049;
            }

            .result {
                margin-top: 25px;
                font-size: 20px;
                font-weight: bold;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>Iris Flower Classifier</h1>

            <p>
                Enter flower measurements
            </p>

            <form method="POST">

                <input
                    type="number"
                    step="any"
                    name="sepal_length"
                    placeholder="Sepal Length"
                    required
                >

                <input
                    type="number"
                    step="any"
                    name="sepal_width"
                    placeholder="Sepal Width"
                    required
                >

                <input
                    type="number"
                    step="any"
                    name="petal_length"
                    placeholder="Petal Length"
                    required
                >

                <input
                    type="number"
                    step="any"
                    name="petal_width"
                    placeholder="Petal Width"
                    required
                >

                <br>

                <button type="submit">
                    Predict Flower
                </button>

            </form>

            <div class="result">

                <p>
                    PREDICTION_PLACEHOLDER
                </p>

                <p>
                    CONFIDENCE_PLACEHOLDER
                </p>

            </div>

        </div>

    </body>

    </html>
    '''

    html = html.replace(
        "PREDICTION_PLACEHOLDER",
        prediction
    )

    html = html.replace(
        "CONFIDENCE_PLACEHOLDER",
        confidence_text
    )

    return html


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
