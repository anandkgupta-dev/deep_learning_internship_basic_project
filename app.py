"""
app.py
======
Streamlit Web Application: Weather Condition Classification Using a Basic Neural Network

Author: B.Tech CSE (AI & ML) Student Project
Course Reference: Introductory Deep Learning (Infosys Springboard)

Technologies: Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Streamlit
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="Weather Classification - Basic ANN",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "weather_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
EVAL_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
HISTORY_IMG_PATH = os.path.join(MODELS_DIR, "training_history.png")
CM_IMG_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
CSV_PATH = os.path.join(DATA_DIR, "weather_data.csv")

CLASS_NAMES = ["Sunny", "Cloudy", "Rainy"]
CLASS_ICONS = {
    "Sunny": "☀️",
    "Cloudy": "⛅",
    "Rainy": "🌧️"
}
CLASS_COLORS = {
    "Sunny": "#f39c12",
    "Cloudy": "#7f8c8d",
    "Rainy": "#2980b9"
}

# -------------------------------------------------------------
# Helper: Cached Model and Scaler Loader
# -------------------------------------------------------------
@st.cache_resource
def load_trained_artifacts():
    """
    Loads the saved Keras neural network model and Scikit-Learn StandardScaler.
    Cached so it does not reload on every user interaction.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    
    # Delayed import of tensorflow to speed up initial UI render
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_evaluation_metrics():
    """Loads actual test evaluation results saved by train_model.py."""
    if not os.path.exists(EVAL_PATH):
        return None
    with open(EVAL_PATH, "r") as f:
        return json.load(f)

# -------------------------------------------------------------
# Main Application Header & Introduction
# -------------------------------------------------------------
st.title("⛅ Weather Condition Classification")
st.subheader("Educational Demonstration Using a Basic Artificial Neural Network (ANN)")

# Status disclaimer alert
st.info(
    "🎓 **College Student Project Notice & Educational Disclaimer:**\n\n"
    "This project is an **educational demonstration** built for an introductory Deep Learning coursework. "
    "It uses a **synthetic, reproducible dataset** to demonstrate neural network concepts such as feature scaling, "
    "hidden layers, ReLU activation, Softmax probability distribution, loss minimization, and epochs. "
    "\n\n⚠️ **Important:** Temperature and relative humidity alone are *insufficient* to reliably predict actual atmospheric weather. "
    "Real-world forecasting requires barometric pressure, wind vectors, dew point, satellite radar, and complex numerical models. "
    "The predictions here reflect model patterns learned from its training data, not certified weather forecasts."
)

# -------------------------------------------------------------
# Check if Model Artifacts Exist
# -------------------------------------------------------------
model, scaler = load_trained_artifacts()
eval_data = load_evaluation_metrics()

if model is None or scaler is None:
    st.error(
        "❌ **Trained Model Files Not Found!**\n\n"
        "The model (`models/weather_model.keras`) or scaler (`models/scaler.joblib`) has not been generated yet.\n\n"
        "**To fix this:** Open your terminal and run the training pipeline first:\n"
        "```bash\n"
        "python train_model.py\n"
        "```\n"
        "Once training completes, refresh this page."
    )
    st.stop()

# -------------------------------------------------------------
# Sidebar: Project Info & Technology Stack
# -------------------------------------------------------------
with st.sidebar:
    st.header("📌 Project Details")
    st.markdown("""
    - **Student:** B.Tech CSE (AI & ML)
    - **Course:** Intro Deep Learning (Infosys Springboard)
    - **Domain:** Multiclass Tabular Classification
    - **Dataset:** Synthetic Educational Dataset (1,000 samples)
    """)
    
    st.divider()
    st.header("🛠️ Technology Stack")
    st.markdown("""
    - **Deep Learning:** TensorFlow / Keras Sequential API
    - **Data Preprocessing:** Scikit-Learn `StandardScaler`
    - **Numerical Computation:** NumPy & Pandas
    - **Visualization:** Matplotlib
    - **Interactive Frontend:** Streamlit
    """)
    
    st.divider()
    st.header("🧠 Network Architecture")
    st.markdown("""
    - **Input Layer:** 2 Features (`Temp`, `Humidity`)
    - **Hidden Layer:** 8 Neurons (`ReLU` activation)
    - **Output Layer:** 3 Neurons (`Softmax` activation)
    - **Loss Function:** Sparse Categorical Cross-Entropy
    - **Optimizer:** Adam (LR = 0.01)
    - **Epochs:** 50
    """)

# -------------------------------------------------------------
# Tabs Layout: 1. Live Prediction, 2. Model Performance, 3. Architecture & Concepts
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Live Weather Prediction", "📊 Model Performance & Metrics", "📚 Deep Learning Concepts & Architecture"])

# =============================================================
# TAB 1: Live Prediction
# =============================================================
with tab1:
    st.write("### Test the Trained Neural Network with New Inputs")
    st.write("Adjust the temperature and humidity below, then click **Predict Weather Condition** to see how the neural network calculates class probabilities.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌡️ Feature 1: Temperature")
        temp_input = st.number_input(
            "Temperature in Celsius (°C):",
            min_value=-15.0,
            max_value=55.0,
            value=28.0,
            step=0.5,
            help="Typical values range from 10°C to 40°C in our training dataset."
        )
        # Visual temperature gauge feedback
        if temp_input < 15:
            st.caption("❄️ Cool / Cold temperature range")
        elif temp_input <= 28:
            st.caption("🌤️ Mild / Moderate temperature range")
        else:
            st.caption("🔥 Warm / Hot temperature range")

    with col2:
        st.markdown("#### 💧 Feature 2: Relative Humidity")
        humidity_input = st.slider(
            "Relative Humidity (%):",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help="Relative humidity measures moisture content in the air (0% = dry, 100% = saturated)."
        )
        if humidity_input < 45:
            st.caption("🏜️ Dry air (common on clear sunny days)")
        elif humidity_input <= 75:
            st.caption("☁️ Moderate humidity (common on overcast/cloudy days)")
        else:
            st.caption("🌧️ Saturated / High moisture (common during rainfall)")

    predict_button = st.button("🚀 Predict Weather Condition", type="primary", use_container_width=True)

    if predict_button:
        # Step 1: Input Validation
        if temp_input < -20.0 or temp_input > 60.0 or humidity_input < 0.0 or humidity_input > 100.0:
            st.error("Input values are outside realistic physical boundaries. Please enter valid meteorological values.")
        else:
            # Step 2: Preprocess using fitted StandardScaler (Feature Scaling)
            input_raw = np.array([[temp_input, humidity_input]], dtype=np.float32)
            input_scaled = scaler.transform(input_raw)

            # Step 3: Neural Network Forward Pass (Inference)
            # Softmax outputs sum to 1.0 (probabilities for Sunny, Cloudy, Rainy)
            pred_probs = model.predict(input_scaled, verbose=0)[0]
            predicted_class_idx = int(np.argmax(pred_probs))
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            confidence = float(pred_probs[predicted_class_idx] * 100)

            # Step 4: Display Primary Prediction
            st.divider()
            st.markdown("### 🎯 Classification Result")
            
            icon = CLASS_ICONS[predicted_class_name]
            st.success(
                f"### Predicted Condition: **{icon} {predicted_class_name}**  \n"
                f"**Model Confidence:** **{confidence:.1f}%**"
            )

            # Step 5: Softmax Probability Distribution Breakdown
            st.markdown("#### 📊 Model Softmax Output Probabilities:")
            st.write("The output layer applies the `Softmax` mathematical function: $\\sigma(z)_i = \\frac{e^{z_i}}{\\sum e^{z_j}}$, producing probabilities across all 3 classes:")

            prob_cols = st.columns(3)
            for i, name in enumerate(CLASS_NAMES):
                prob_pct = float(pred_probs[i] * 100)
                with prob_cols[i]:
                    st.metric(
                        label=f"{CLASS_ICONS[name]} {name}",
                        value=f"{prob_pct:.1f}%"
                    )
                    st.progress(float(pred_probs[i]))

            # Step 6: Step-by-Step Educational Explanation of the Prediction
            with st.expander("🔍 See Exact Step-by-Step Mathematical Flow for this Input"):
                st.markdown(f"""
                1. **Raw User Input:**
                   - Temperature = `{temp_input} °C`
                   - Humidity = `{humidity_input} %`
                2. **StandardScaler Transformation:**
                   - Fitted Scaler Means: `Temp Mean = {scaler.mean_[0]:.2f}°C`, `Humidity Mean = {scaler.mean_[1]:.2f}%`
                   - Fitted Scaler Scales (std): `Temp Std = {scaler.scale_[0]:.2f}`, `Humidity Std = {scaler.scale_[1]:.2f}`
                   - Scaled Formula: $z = \\frac{{x - \\mu}}{{\\sigma}}$
                   - **Scaled Input Vector passed to ANN:** `[{input_scaled[0][0]:.4f}, {input_scaled[0][1]:.4f}]`
                3. **Hidden Layer (8 Neurons, ReLU):**
                   - Computes linear combination: $Z_1 = X \\cdot W_1 + b_1$
                   - Applies ReLU activation: $A_1 = \\max(0, Z_1)$ (replaces negative activations with 0).
                4. **Output Layer (3 Neurons, Softmax):**
                   - Computes logits: $Z_2 = A_1 \\cdot W_2 + b_2$
                   - Applies Softmax activation: transforms 3 raw values into probabilities summing to exactly 1.0.
                5. **Final ArgMax Selection:**
                   - Highest probability corresponds to index `{predicted_class_idx}` -> **{predicted_class_name}** ({confidence:.1f}%).
                """)

# =============================================================
# TAB 2: Model Performance & Actual Evaluation Results
# =============================================================
with tab2:
    st.write("### 📈 Actual Model Evaluation Results on Unseen Test Dataset")
    st.write(
        "These results are loaded directly from `models/evaluation_results.json` produced by the "
        "`train_model.py` pipeline. They reflect honest evaluation on a held-out 20% test dataset."
    )

    if eval_data:
        # Display High-Level Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{eval_data.get('test_accuracy_pct', 0.0)}%")
        m2.metric("Test Loss", f"{eval_data.get('test_loss', 0.0)}")
        m3.metric("Test Samples", f"{eval_data.get('total_test_samples', 0)}")
        m4.metric("Training Epochs", f"{eval_data.get('training_epochs', 50)}")

        st.divider()

        # Display Plots (Training History and Confusion Matrix)
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("#### 📉 Learning Curves (Loss & Accuracy vs. Epochs)")
            if os.path.exists(HISTORY_IMG_PATH):
                st.image(HISTORY_IMG_PATH, caption="Training & Validation Loss and Accuracy over 50 Epochs", use_container_width=True)
            else:
                st.warning("Training history plot image not found.")

        with p_col2:
            st.markdown("#### 🔲 Confusion Matrix on Held-Out Test Set")
            if os.path.exists(CM_IMG_PATH):
                st.image(CM_IMG_PATH, caption="Confusion Matrix: True Labels vs. Model Predicted Labels", use_container_width=True)
            else:
                st.warning("Confusion matrix plot image not found.")

        st.divider()

        # Classification Report Table
        st.markdown("#### 📋 Detailed Classification Report (Precision, Recall, F1-Score)")
        cr = eval_data.get("classification_report", {})
        
        # Prepare dataframe for table display
        table_rows = []
        for c in CLASS_NAMES:
            if c in cr:
                table_rows.append({
                    "Class": c,
                    "Precision": f"{cr[c]['precision']:.2f}",
                    "Recall": f"{cr[c]['recall']:.2f}",
                    "F1-Score": f"{cr[c]['f1-score']:.2f}",
                    "Test Samples (Support)": int(cr[c]['support'])
                })
        if table_rows:
            report_df = pd.DataFrame(table_rows)
            st.table(report_df)
            
        st.markdown("""
        **Metric Explanations for College Viva:**
        - **Precision:** Out of all times the model predicted this class, what fraction was actually correct?
        - **Recall:** Out of all actual samples of this class in the test set, what fraction did the model successfully find?
        - **F1-Score:** The harmonic mean of Precision and Recall: $\\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$.
        """)
    else:
        st.warning("Evaluation metrics file `models/evaluation_results.json` not found.")

# =============================================================
# TAB 3: Deep Learning Concepts & Architecture (Viva Prep)
# =============================================================
with tab3:
    st.write("### 📖 Core Deep Learning Concepts Explained Simply")
    st.write("Use this reference to prepare your answers for your lab viva or presentation.")

    with st.expander("1. Why do we need Feature Scaling (`StandardScaler`)?"):
        st.write("""
        - **The Problem:** In our dataset, Temperature is typically in the range `10°C to 40°C`, while Humidity is `15% to 100%`.
        - Because Humidity has larger numerical values, unscaled weights for Humidity would dominate weight updates during gradient descent.
        - **The Solution:** `StandardScaler` standardizes each feature to have a mean of 0 and standard deviation of 1 using the formula:
          $$z = \\frac{x - \\mu}{\\sigma}$$
        - **Crucial Rule:** We fit the scaler **only** on the training set (`scaler.fit(X_train)`) and then transform both train and test sets (`transform`). This prevents **data leakage**!
        """)

    with st.expander("2. Why 8 Hidden Neurons and ReLU Activation?"):
        st.write("""
        - **Small, Explainable Size:** 8 neurons are sufficient to learn non-linear boundaries between 3 classes with 2 inputs without overfitting.
        - **ReLU (Rectified Linear Unit):**
          $$f(x) = \\max(0, x)$$
        - If input is negative, ReLU outputs `0`. If positive, it outputs `x`.
        - Without an activation function, stacking multiple layers would just equal a single linear regression. ReLU introduces non-linearity, allowing the network to capture complex decision curves.
        """)

    with st.expander("3. Why Softmax Activation on the Output Layer?"):
        st.write("""
        - The output layer has **3 neurons**, representing Sunny, Cloudy, and Rainy.
        - The **Softmax** activation function converts 3 arbitrary real numbers (called logits) into valid probabilities that:
          1. Are all between 0.0 and 1.0.
          2. Sum to exactly 1.0 (100%).
        - The class with the highest probability is chosen using `argmax`.
        """)

    with st.expander("4. Why Sparse Categorical Cross-Entropy Loss?"):
        st.write("""
        - **Loss** measures the difference (error) between the model's predicted probability distribution and the actual true label.
        - We use **Sparse** Categorical Cross-Entropy because our true labels are encoded as integers (`0 = Sunny`, `1 = Cloudy`, `2 = Rainy`) rather than one-hot encoded vectors (`[1, 0, 0]`, etc.).
        - It mathematically penalizes the model when it assigns low probability to the correct class.
        """)

    with st.expander("5. What are Epochs and Batch Size?"):
        st.write("""
        - **Epoch:** One complete pass through the entire training dataset. In our training, `epochs = 50`.
        - **Batch Size:** The number of training examples processed before updating the neural network weights. We use `batch_size = 16`.
        - For 640 training samples with batch size 16, there are $\\frac{640}{16} = 40$ weight update steps per epoch.
        """)

    with st.expander("6. Why Can't Temperature and Humidity Alone Reliably Forecast Weather?"):
        st.write("""
        - **Scientific Limitation:** Real meteorology is a dynamic physical science governed by fluid dynamics and thermodynamics.
        - Real forecasts require atmospheric pressure, wind direction and speed, dew point, cloud cover altitude, solar radiation, and historical pressure trends.
        - Two simple scalar measurements cannot differentiate between an impending summer thunderstorm and a humid overcast day with 100% accuracy.
        - This demonstrates the honesty required in engineering: understanding model limitations is just as important as measuring accuracy!
        """)

st.divider()
st.caption("Built with TensorFlow/Keras & Streamlit • Educational Deep Learning Project • B.Tech CSE (AI & ML)")
