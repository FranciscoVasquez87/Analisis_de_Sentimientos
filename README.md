📊 Proyecto Final de Machine Learning: Clasificador de Sentimiento de Reseñas de Google Play Store

Visión General del Proyecto

Este proyecto final aplica técnicas de Machine Learning y Procesamiento de Lenguaje Natural (NLP) para construir y desplegar un modelo capaz de clasificar automáticamente el sentimiento (Positivo, Negativo o Neutral) de las reseñas de la Google Play Store.

El objetivo principal es proveer una herramienta de API REST que permita a desarrolladores o analistas de mercado obtener insights instantáneos sobre la percepción de los usuarios, sin necesidad de leer miles de comentarios manualmente.

🎯 Problema a Resolver

El dataset de Google Play Store Apps Reviews contiene una gran cantidad de texto libre (reseñas). Clasificar estas reseñas manualmente es ineficiente. El reto es crear un sistema automatizado y preciso que categorice cada reseña en uno de tres sentimientos:

Positive (Positivo)

Negative (Negativo)

Neutral (Neutral)

🛠️ Metodología de Solución

1. Análisis Exploratorio de Datos (EDA) y Limpieza (Actividad 1 - 20%)

Completitud de Datos: Se identificaron y manejaron los valores faltantes en la columna de sentimiento para asegurar la calidad del target.

Limpieza de Texto: Se aplicó un proceso de limpieza de las reseñas que incluyó:

Conversión a minúsculas.

Eliminación de puntuación y dígitos.

Tokenización y, si fue necesario, manejo de stopwords y lematización (ajustar según tu caso real).

Visualización: Se generaron nubes de palabras y gráficos de distribución para entender el corpus y la proporción de cada sentimiento.

2. Procesamiento y Entrenamiento del Modelo (Actividad 2 y 3 - 40%)

Paso

Técnica/Modelo

Propósito

Vectorización

TF-IDF

Transformar el texto limpio en características numéricas (vectores de peso) para que el modelo pueda interpretarlas.

Modelo Base

SVC (Soporte Vectorial) / Random Forest

Se probó un modelo base, eligiendo Random Forest Classifier como el clasificador final por su robustez en problemas multiclase.

Ensembling

Random Forest

El modelo es en sí un método de ensamble (bagging), lo que ayuda a reducir la varianza y aumentar la precisión general.

Tuning

GridSearchCV

Se optimizaron los hiperparámetros del Random Forest (usando métrica f1_macro) para mejorar el rendimiento.

Mejores Hiperparámetros (Grid Search):

'criterion': 'gini'

'max_depth': None

'min_samples_leaf': 1

'n_estimators': 200

3. Métricas de Rendimiento (Actividad 3 - 20%)

Métrica

Puntuación

Explicación Coloquial

Accuracy Global

0.9120 (91.20%)

De cada 100 reseñas que probamos, el modelo predice correctamente el sentimiento de aproximadamente 91 de ellas.

F1-Score (Macro)

0.8858 (88.58%)

Esta métrica es la media equilibrada entre precisión y recall para todas las clases (Positivo, Negativo, Neutral). Un valor alto como este indica que el modelo no solo es bueno para la clase mayoritaria, sino que también tiene un buen desempeño prediciendo las clases menos frecuentes (Negativo y Neutral).

Visualización de Rendimiento

Se incluye una Matriz de Confusión visualizada con Seaborn (cmap='viridis') para ilustrar cómo el modelo clasifica correctamente y dónde se confunde.

El 94% de las reseñas Positivas fueron identificadas correctamente, y el modelo mostró una gran capacidad para distinguir reseñas Neutrales y Negativas. (Ajusta los porcentajes según tu matriz real si difieren ligeramente).

4. Construcción y Despliegue de la API REST (Actividad 4 - 20%)

La solución se implementa como un servicio web utilizando el framework Flask en Python.

Componentes de la API:

Archivos Serializados (.pkl): Se cargan el modelo (best_rf_model.pkl), el vectorizador TF-IDF (tfidf_vectorizer.pkl) y el codificador de etiquetas (label_encoder.pkl).

app.py: Contiene la lógica del servidor Flask y el endpoint /predict.

Uso del Endpoint /predict

Detalle

Especificación

Método HTTP

POST

Endpoint

/predict

Cuerpo de la Solicitud

JSON con la clave review.

Ejemplo de Cuerpo

{"review": "El último parche de la app ha arreglado todos los errores, genial!"}

Respuesta de la API (JSON):

{
  "status": "success",
  "review_input": "El último parche de la app ha arreglado todos los errores, genial!",
  "prediction": "Positive",
  "confidence_percentage": "98.50%"
}


🚀 Cómo Ejecutar y Probar la API Localmente

Requisitos: Asegúrate de tener Python instalado.

Archivos Necesarios:

best_rf_model.pkl

tfidf_vectorizer.pkl

label_encoder.pkl

requirements.txt

app.py

Instalar Dependencias: Abre tu terminal en la carpeta del proyecto y ejecuta:

pip install -r requirements.txt


Ejecutar la API:

python app.py


La API estará disponible, por defecto, en http://127.0.0.1:5000/.

Prueba con Postman/API Test:

Método: POST

URL: http://127.0.0.1:5000/predict

Pestaña: Body -> raw -> JSON

Cuerpo de la Prueba:

{
    "review": "Esta actualización es terrible, la aplicación ahora se bloquea constantemente."
}


Deberías recibir la respuesta JSON con la predicción.
