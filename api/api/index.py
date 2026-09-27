from flask import Flask, request
import numpy as np
import os

app = Flask(__name__)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

model_path = os.path.join(
    BASE_DIR,
    "iris_mlp_model.npz"
)

model = np.load(model_path)

W1 = model["W1"]
b1 = model["b1"]

W2 = model["W2"]
b2 = model["b2"]

W3 = model["W3"]
b3 = model["b3"]

mean = model["mean"]
scale = model["scale"]

classes = [
    "Setosa",
    "Versicolor",
    "Virginica"
]


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def relu(x):
    return np.maximum(0, x)


def softmax(x):

    exp_values = np.exp(
        x - np.max(
            x,
            axis=1,
            keepdims=True
        )
    )

    return (
        exp_values /
        np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_flower(features):

    X = np.array(
        features,
        dtype=float
    )

    # Standardization
    X = (X - mean) / scale

    X = X.reshape(1, -1)


    # Hidden Layer 1
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)


    # Hidden Layer 2
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)


    # Output Layer
    Z3 = np.dot(A2, W3) + b3

    probabilities = softmax(Z3)


    predicted_class = np.argmax(
        probabilities,
        axis=1
    )[0]


    confidence = (
        probabilities[0][predicted_class]
        * 100
    )


    return (
        classes[predicted_class],
        confidence
    )


# ==========================================
# HOME PAGE
# ==========================================

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

<title>Iris Flower Classifier</title>

<style>

body {
    font-family: Arial;
    background: #f2f2f2;
    text-align: center;
    padding: 50px;
}

.container {
    background: white;
    max-width: 550px;
    margin: auto;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.15);
}

h1 {
    margin-bottom: 10px;
}

p {
    color: #666;
}

input {
    width: 80%;
    padding: 12px;
    margin: 8px;
    border: 1px solid #ccc;
    border-radius: 6px;
    font-size: 15px;
}

button {
    padding: 12px 30px;
    border: none;
    border-radius: 6px;
    background: #333;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #555;
}

.result {
    margin-top: 25px;
    font-size: 24px;
    font-weight: bold;
}

.confidence {
    margin-top: 10px;
    font-size: 18px;
}

</style>

</head>


<body>

<div class="container">

<h1>🌸 Iris Flower Classifier</h1>

<p>
Multilayer Perceptron Neural Network
</p>


<form method="POST">


<input
type="number"
step="any"
name="sepal_length"
placeholder="Sepal Length (cm)"
required
>

<br>


<input
type="number"
step="any"
name="sepal_width"
placeholder="Sepal Width (cm)"
required
>

<br>


<input
type="number"
step="any"
name="petal_length"
placeholder="Petal Length (cm)"
required
>

<br>


<input
type="number"
step="any"
name="petal_width"
placeholder="Petal Width (cm)"
required
>

<br><br>


<button type="submit">
Predict Flower
</button>


</form>


<div class="result">
PREDICTION_PLACEHOLDER
</div>


<div class="confidence">
CONFIDENCE_PLACEHOLDER
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
"""
