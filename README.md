# modelos_clasificacion_scoring_credito
Predicción de Riesgo de Impago en Préstamos P2P — Prosper Loan Data

Proyecto de Machine Learning que compara cinco modelos de clasificación para estimar el riesgo de impago de un préstamo al momento de su originación.

1. Problema
Dominio: scoring crediticio en plataformas de préstamos P2P (peer-to-peer), usando datos de Prosper.
Objetivo: predecir el riesgo de impago de un solicitante antes de aprobar el crédito, usando únicamente variables disponibles al momento de la solicitud (no variables de desempeño posterior al desembolso).
Tipo de problema: clasificación. Variable objetivo (LoanStatus_num) con 3 clases: Pago cumplido, Mora, Default.
Utilidad: apoyar decisiones de originación de crédito, priorizando la identificación de solicitantes de alto riesgo.

2. Dataset
Fuente: Prosper Loan Data (Kaggle).
Tamaño: 113.937 préstamos, 81 variables originales.
Variables finales del modelo: 25 numéricas + 7 categóricas, tras un proceso de limpieza:
FirstRecordedCreditLine (fecha, alta cardinalidad) se transformó en AntiguedadCrediticiaAnios (numérica).
Se eliminaron variables redundantes o casi constantes: ProsperRating (Alpha) (duplica ProsperRating (numeric)), CreditGrade (sistema legacy pre-2009, fuertemente imputado), Recommendations (~96% en un solo valor) y LoanOriginationQuarter (efecto temporal, alta cardinalidad).
Las categorías poco frecuentes de Occupation y BorrowerState se agruparon en "Otras" (umbral 1%) para reducir la dimensionalidad tras el one-hot encoding.
Desbalance de clases: Pago cumplido 83.3% / Default 14.9% / Mora 1.8%.

3. Solución propuesta
EDA: descripción de cada variable (tipo, rango, escala), análisis de desbalance y matriz de correlación entre variables numéricas.
Preprocesamiento: StandardScaler (numéricas) + OneHotEncoder (categóricas) vía ColumnTransformer.
Validación: StratifiedKFold de 10 folds, con F1-macro como métrica principal (adecuada para el fuerte desbalance de clases).
Modelos entrenados, cada uno con búsqueda de hiperparámetros (GridSearchCV):
Árbol de Decisión
Bagging (árboles)
Random Forest
AdaBoost
XGBoost

4. Resultado
Mejor modelo: Random Forest.

Experimento adicional: reformulación binaria

Como análisis complementario, se colapsaron las clases Mora y Default en una única clase Impago, replanteando el problema como clasificación binaria (Pago cumplido vs. Impago). Bajo esta formulación, el F1-macro mejora sustancialmente en todos los modelos (Random Forest: 0.6864), lo que confirma que buena parte del bajo desempeño multiclase se debía al fuerte desbalance y a la dificultad de separar Mora de Default.

5. Conclusiones
   
- Random Forest fue el modelo con mejor desempeño (F1-macro de 0.47), gracias a que combina variedad de datos y de variables entre sus árboles. Su principal desventaja frente a un árbol individual o AdaBoost es el mayor costo computacional, al entrenar cientos de árboles en vez de uno.
- El desempeño moderado-bajo de los cinco modelos en la formulación multiclase se explica en gran parte por el fuerte desbalance del dataset: con apenas 1.8% de los casos en la clase Mora, los modelos no logran distinguirla del resto (F1 ≈ 0.07 en esa clase, frente a 0.70 en Pago cumplido), lo que arrastra el F1-macro promedio hacia abajo.
- Al reducir el problema a dos clases (Pago cumplido e Impago), los F1 mejoraron sustancialmente frente al problema con tres clases. Esto confirma que buena parte de la dificultad original venía del desbalance y de la fuerte similitud entre Mora y Default —dos grados del mismo fenómeno de incumplimiento—, que los modelos confundían constantemente entre sí. Parte de esta mejora también se explica porque un problema binario es, por su propia naturaleza, más fácil de puntuar que uno de tres clases: no todo el salto refleja una mejora real del modelo.
- Los resultados sugieren que separar Mora y Default como clases distintas aporta poco valor práctico. Para un banco, la pregunta operativamente relevante suele ser binaria —¿paga o no paga?—, por lo que un enfoque de dos clases resulta más adecuado que intentar sostener una distinción de tres clases que se demostró difícil de separar incluso para el mejor modelo evaluado.

