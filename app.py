import flask
from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
import re
from functools import wraps

# --- Configuración y Carga de Componentes ---

app = Flask(__name__)

# Rutas de los archivos .pkl (deben estar en la misma carpeta que app.py)
MODEL_PATH = 'best_rf_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# Cargar componentes. Usamos joblib para el modelo y pickle para los demás.
try:
    # Nota: Es crucial importar pandas (aunque no se use directamente para la predicción)
    # porque joblib/pickle a veces necesitan las librerías usadas para serializar los objetos.
    import pandas as pd # Importado aquí para la función clean_text
    best_rf_model = joblib.load(MODEL_PATH)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Modelos y componentes cargados exitosamente.")
except FileNotFoundError:
    print(f"ERROR: No se pudieron encontrar los archivos .pkl. Asegúrate de que '{MODEL_PATH}', '{VECTORIZER_PATH}' y '{ENCODER_PATH}' estén en la misma carpeta.")
    best_rf_model = None
    vectorizer = None
    label_encoder = None
except Exception as e:
    print(f"Error al cargar los componentes: {e}")
    best_rf_model = None
    vectorizer = None
    label_encoder = None

# --- Función de Preprocesamiento CRÍTICA ---
# ESTA FUNCIÓN DEBE SER IDÉNTICA a la que usaste en tu EDA para crear 'review_clean'.
def clean_text(text):
    # Nota: Se usa pd.isna si pandas está importado, si no, se usa una verificación básica
    if text is None or (hasattr(pd, 'isna') and pd.isna(text)):
        return ""
    # Convertir a minúsculas
    text = str(text).lower()
    # Eliminar signos de puntuación
    text = re.sub(r'[^\w\s]', '', text)
    # Eliminar dígitos
    text = re.sub(r'\d+', '', text)
    # Quitar espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Decorador para verificar que el modelo esté cargado
def model_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if best_rf_model is None:
            return jsonify({
                "error": "Error Interno del Servidor",
                "message": "Los componentes del modelo no se pudieron cargar. Verifique la existencia de los archivos .pkl."
            }), 500
        return f(*args, **kwargs)
    return decorated_function

# --- Ruta Principal para la Predicción ---

@app.route('/predict', methods=['POST'])
@model_required
def predict():
    """
    Endpoint que recibe una reseña de Google Play y devuelve el sentimiento predicho
    junto con el porcentaje de confianza.
    Ejemplo de JSON de entrada: {"review": "Esta aplicación es fantástica, me encanta."}
    """
    try:
        # 1. Obtener los datos del cuerpo de la solicitud
        data = request.get_json(force=True)
        raw_review = data.get('review', '')

        if not raw_review:
            return jsonify({
                "error": "Entrada inválida",
                "message": "La solicitud POST debe contener un campo 'review' con el texto a analizar."
            }), 400

        # 2. Preprocesamiento
        cleaned_review = clean_text(raw_review)
        
        # 3. Vectorización (usando el vectorizador ajustado)
        # Nota: transform() en el vectorizador, NO fit_transform()
        review_vec = vectorizer.transform([cleaned_review])
        
        # 4. Predicción (etiqueta y probabilidad)
        # El modelo espera una entrada vectorizada
        
        # Predecir la etiqueta codificada (0, 1, 2)
        y_pred_encoded = best_rf_model.predict(review_vec)
        # Predecir las probabilidades (para la confianza)
        y_pred_proba = best_rf_model.predict_proba(review_vec)
        
        # 5. Transformación inversa y cálculo de confianza
        
        # Convertir la etiqueta codificada a la etiqueta original (e.g., 'Positive')
        predicted_sentiment = label_encoder.inverse_transform(y_pred_encoded)[0]
        
        # Obtener la probabilidad de la clase predicha para la confianza
        # Se necesita el índice de la clase predicha
        predicted_class_index = y_pred_encoded[0]
        confidence_score = y_pred_proba[0, predicted_class_index]

        # 6. Devolver el resultado
        return jsonify({
            "status": "success",
            "review_input": raw_review,
            "prediction": predicted_sentiment,
            "confidence_percentage": f"{confidence_score * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({
            "error": "Error durante la predicción",
            "details": str(e)
        }), 500

# --- Ejecución del Servidor ---
if __name__ == '__main__':
    # Ejecuta el servidor en el puerto 5000 por defecto
    # En producción (como Netlify o Ngrok) debes usar el puerto que te indiquen.
    app.run(debug=True)
