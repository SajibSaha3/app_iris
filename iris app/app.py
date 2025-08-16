import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================
# Function to train model
# =========================
def train_iris_model():
    # Load dataset
    df = sns.load_dataset("iris")

    # Split features and target
    X = df.drop(["species"], axis=1)
    y = df["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Accuracy
    acc = model.score(X_test, y_test)

    return model, acc

# =========================
# Streamlit App
# =========================
st.title("🌸 Iris Flower Prediction App")

# Train the model
model, accuracy = train_iris_model()
st.write(f"✅ Model trained with Test Accuracy: **{accuracy:.2f}**")

st.write("### Enter flower measurements:")

# User inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create input DataFrame
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    species_map = {0: "Setosa 🌱", 1: "Versicolor 🌸", 2: "Virginica 🌺"}
    st.success(f"### Predicted Species: **{species_map[prediction]}**")

    # Probability
    proba = model.predict_proba(input_data)
    st.write("#### Prediction Probabilities:")
    st.dataframe(pd.DataFrame(proba, columns=species_map.values()))
