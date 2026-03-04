# **Challenge3_TelecomX_LATAM Parte2- grupo G9**

___________________________________________________________________________

**Desarrollado : Jaime Pradenas **
** Febrero 2026 **

- Plantilla Trello - https://trello.com/b/uH4K5Eek/telecomxlatamparte2jpraden
- Repositorio Github - https://github.com/jpraden/Challenge-DataScience-TelecomX-LATAM_Parte2

💡**Acerca del desafío**💡
**Telecom X – Parte 2: Predicción de Cancelación (Churn)**


**Descripción**

📣 Historia del Desafío

En la etapa anterior, se ha realizado el análisis exploratorio de la cancelación de clientes en Telecom X Parte1, bajo el rol "Asistente de análisis de datos", donde fue entregado con claridad los resultados y visión estratégica marcaron la diferencia.

**Ahora, ¡he sido invitado oficialmente a formar parte del equipo de Machine Learning de la empresa!**

____________________________________________________________________________

🎯 Misión

Es desarrollar modelos predictivos capaces de prever qué clientes tienen mayor probabilidad de cancelar sus servicios.

La empresa quiere anticiparse al problema de la cancelación, y en este rol corresponde construir un pipeline robusto para esta etapa inicial de modelado.

_____________________________________________________________________________

🧠 Objetivos del Desafío

Preparar los datos para el modelado (tratamiento, codificación, normalización).

Realizar análisis de correlación y selección de variables.

Entrenar dos o más modelos de clasificación.

Evaluar el rendimiento de los modelos con métricas.

Interpretar los resultados, incluyendo la importancia de las variables.

Crear una conclusión estratégica señalando los principales factores que influyen en la cancelación.

_______________________________________________________________________________

🧰 Lo que se debe aplicar, bajo el rol **"Analista Junior de Machine Learning"**

    ✅ Preprocesamiento de datos para Machine Learning

    ✅ Construcción y evaluación de modelos predictivos

    ✅ Interpretación de resultados y entrega de insights

    ✅ Comunicación técnica con enfoque estratégico

___________________________________________________________________________________________________________

🛠️ Tecnologías y Herramientas
- Trello 
- Github
- Google Colab - Entorno de desarrollo (Python - Librerias usadas)
  * Pandas – Manipulación y análisis de datos.
  * Matplotlib – Visualización de datos.
  * Seaborn – Visualizaciones estadísticas.
  * NumPy – Operaciones numéricas.
  * sklearn – Librería (paquete) estándar en Python para machine learning
      * import LabelEncoder - está en el submódulo sklearn.preprocessing, Se usa para transformar etiquetas categóricas en valores numéricos.
      * import chi2 - está en el submódulo sklearn.feature_selection, útil para evaluar la relación entre variables independientes y la variable objetivo.
      * import train_test_split - es una función que está en la librería scikit-learn para (entrenamiento y prueba),
      * import LogisticRegression - se encuentra en el submódulo sklearn.linear_model. usado como clasificador lineal para problemas de clasificación binaria o multiclase.
      * import classification_report - se encuentra en el submódulo sklearn.metrics entrega un resumen de métricas de clasificación,
      * import RandomForestClassifier -  proviene de la librería scikit-learn, específicamente del submódulo sklearn.ensemble que combina muchos árboles de decisión para lograr un modelo más robusto y preciso.
      * import StandardScaler - proviene de la librería scikit-learn, específicamente del submódulo sklearn.preprocessing, se conoce como normalización estándar (z-score scaling) y se usa para escalar variables numéricas y mejorar el rendimiento de modelos sensibles a la escala.
      * import confusion_matrix - proviene de la librería scikit-learn, específicamente del submódulo sklearn.metrics,  una herramienta fundamental para evaluar el rendimiento de modelos de clasificación.
  *  imbalanced-learn - Librería (abreviada como imblearn), que es un paquete complementario a scikit-learn diseñado para trabajar con datasets desbalanceados.
      * import SMOTE - - SMOTE (Synthetic Minority Over-sampling Technique) es una técnica para balancear datasets desbalanceados.

_____________________________________________________________________________________________________________________________________________


📦 Instalación y Configuración
- Prerrequisitos Python 3.8 o superior
  * pandas>=1.5.0
  * matplotlib>=3.5.0
  * seaborn>=0.12.0
  * numpy>=1.23.0
  * sklearn>= para machine learning

___________________________________________________________________________________________________________


🛠️ Repositorio del proyecto
- Github --> https://github.com/jpraden/Challenge-DataScience-TelecomX-LATAM_Parte2/TelecomX_LATAM.ipynb

🛠️ Estructura Carpetas en GitHub

../Challenge-DataScience-TelecomX-LATAM_Parte2

    -> TelecomX_LATAM_Parte2
    -> README.md

     ../datos
       -> datos_tratados.csv
     ../modelos
        -> log_model_smote.pkl
        -> scaler.pkl

___________________________________________________________________________________________________________-

🚀 Ejecución del proyecto
- Opción 1: 
  * Abrir cuaderno desde Google Colab (Recomendado)
  * Ejecutar Todo o secuencialmente de nueva casilla

- Opción 2: Jupyter notebook
- Opción 3: Visual Studio Code


___________________________________________________________________________________________________________

📊 #**Actividades realizadas**


🎯##Objetivo 

Identificar los principales factores que afectan la cancelación de clientes y proponer estrategias de retención basadas en los resultados obtenidos. 

___________________________________________________________________

#📌 Extracción del Archivo Tratado

**Descripción:**

Carga el archivo CSV que contiene los datos tratados anteriormente.
📂 Atención: Utiliza el mismo archivo que limpiaste y organizaste en la Parte 1 del desafío Telecom X. Debe contener solo las columnas relevantes, ya con los datos corregidos y estandarizados.

**Acceso a los datos desde API repositorio en GitHub para usar en código:**

🔗 Telecom X --> https://github.com/jpraden/Challenge-DataScience-TelecomX-LATAM_Parte2/datos/datos_tratados.csv
______________________________________________________________________________


**Actividades a realizar en esta primera etapa:**

✅ Importar desde una API ubicada en ruta GitHub, señalada.

✅ Verificar información contiene el dataset (Estructura y datos).

_____________________________________________________________________________

📚 **Bibliotecas y librerias para extracción, exploración y manipulación de datos**
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

________________________________________________________________________________________

**🔍** **Resultado de extracción de datos tratados de la tapa anterior**

*Archivo "datos_tratados.cvs"*

  - **filas** - `'7.032'`: Cantidad de registros.
  - **columnas** - `'22'`: Atributos de los datos.

______________________________________________________________________________


**DICCIONARIO DE DATOS - Dataset**

1. `'id'`: Identificador único del cliente.
2. `'Churn'`: Indica si el cliente canceló el servicio (`Sí` o `No`).
3. `'genero'`: Género del cliente (`Male` o `Female`).
4. `'tiene +60'`: Indica si el cliente es una persona mayor (0 = No, 1 = Sí).
5. `'posee_pareja'`: Si el cliente tiene pareja (`Sí` o `No`).
6. `'posee_dependientes'`: Si el cliente tiene dependientes (`Sí` o `No`).
7. `'tiempo_contrato'`: Tiempo de permanencia como cliente (en meses).
8. `'servicio_telefono'`: Indica si el cliente posee servicio de teléfono (`Yes` o `No`, `Sin servicio de teléfono`).
9. `'multiples_lineas'`: Si posee múltiples líneas telefónicas. (`Sí` o `No`,  `Sin servicio de teléfono`).
10. `'tipo_internet'`: Servicio de internet (`DSL`, `Fiber optic`, `No`).
11. `'seguridad_online'`: Servicio internet de seguridad (`No`, `Sí`, `Sin servicio de internet'`).
12. `'backup_online '`: Servicio internet de respaldo (`No`, `Sí`, `Sin servicio de internet`).
13. `'proteccion_dispositivo'`: Servicio internet proteciòn (`No`, `Sí`, `Sin servicio de internet`).
14. `'soporte_tecnico'`: Servicio internet soporte tècnico (`No`, `Sí`, `Sin servicio de internet`).
15. `'streaming_tv'`: Servicio internet streaming de televisión (`No`, `Sí`, `Sin servicio de internet`).
16. `'streaming_peliculas'`: Servicio internet streaming de peliculas (`No`, `Yes`, `Sin servicio de internet`).
17. `'tipo_contrato'`: Tipo de contrato (`Mensual`, `Anual`, `Bianual`).
18. `'factura_digital'`: Si el cliente recibe facturas electrónicas (`Sí` o `No`).
19. `'metodo_pago'`: Método de pago (`Transferencia bancaria (automática)`, `Tarjeta de crédito (automático)`, `Cheque enviado por correo`, `Cheque electrónico`).
20. `'valor_mensal'`: Valor mensual cobrado.
21. `'total_cobrado'`: Valor total pagado por el cliente.
22. `'cuentas_diarias'`: Valor cargo en cuenta por día.

____________________________________________________________________________

#📌🔧 Preprocesamiento

✔️Eliminando columnas irrelevantes

Eliminando columnas que no aportan valor al análisis o a los modelos predictivos, como identificadores únicos (por ejemplo, el ID del cliente). Estas columnas no ayudan en la predicción de la cancelación y pueden incluso perjudicar el desempeño de los modelos.

- 'id' 
- cuentas_diarias
- total_cobrado

✔️ Ajustando registro atributos (reemplazo del conteno 'Sin servicio de internet' por'No') para:
- 'seguridad_online'
- 'backup_online'
- 'proteccion_dispositivo'
- 'soporte_tecnico'
- 'streaming_tv', 
- 'streaming_peliculas'

✔️ Verificación nuevamente de los valores nulos

_____________________________________________________________________________

✔️  Análisis Dirigido para:


![Visualización](/graficos/Visualización_BoxPlot_CancelacionTiempoPermanencia.png)



🔍 Comparación general
    - El gráfico confirma que existe una relación inversa entre tiempo de contrato y churn: a mayor permanencia, menor probabilidad de cancelación.
    - La diferencia entre las cajas (medianas y rangos intercuartílicos) de ambas categorías es clara y consistente.

**En resumen :** El boxplot evidencia que los clientes con mayor tiempo de permanencia son menos propensos a cancelar, mientras que los nuevos clientes presentan mayor riesgo de churn. Esto sugiere que las estrategias de retención deben enfocarse en los primeros meses de relación, cuando la probabilidad de abandono es más alta.

![Visualización](/graficos/Visualización_BoxPlot_ClientesConCancelacionEtapasTempranas.png)
_______________________________________________________________________________

- Gasto total × Cancelación

![Visualización](/graficos/Visualizacion_CancelacionVSGastosTotales.png)

🔍 Comparación general
    - Existe una relación inversa entre churn y cargos totales: los clientes con menor gasto acumulado son más propensos a cancelar.
    - Esto se alinea con la idea de que los clientes nuevos o de corta permanencia (con menos cargos acumulados) tienen mayor riesgo de abandono.

_______________________________________________________________________________

**En resumen:** El gráfico evidencia que los clientes con cargos totales bajos son más propensos a cancelar, mientras que quienes acumulan mayores pagos tienden a permanecer. Esto sugiere que las estrategias de retención deben enfocarse en clientes de bajo gasto acumulado, ya que representan el grupo con mayor riesgo de churn.

_____________________________________________________________________________

#📊 EDA 

1. Correlación entre variables numericas

![Visualización](/graficos/Visualizacion_CorrelacionVariablesNumericas.png)

 **Resultados de la matriz de correlación (variables numericas)**
- tiempo_contrato: correlación negativa moderada con Churn (-0.35).
  → Clientes con contratos más largos tienen menor probabilidad de cancelar.
  → Es el predictor numérico más relevante.
- valor_mensual: correlación positiva débil con Churn (0.19).
  → Cargos mensuales más altos se asocian ligeramente con mayor cancelación.
- tiene +60: correlación positiva débil con Churn (0.15).
  → Los clientes mayores muestran una leve tendencia a cancelar.


_____________________________________________________________________________

2. Analisis de churn por categoria (variables categoricas)

 - Seleccionar variables categoricas 
    //****************************************************//
    Ranking de variables categóricas respecto a Churn:
    //****************************************************//
                      Variable  Chi2 Score        p-value
    14             metodo_pago  466.119470  2.239792e-103
    12           tipo_contrato  392.706273   2.131687e-87
    6         seguridad_online  147.165601   7.219883e-34
    9          soporte_tecnico  135.439602   2.645952e-31
    2       posee_dependientes  131.271509   2.159540e-30
    13         factura_digital  104.979224   1.234232e-24
    1             posee_pareja   81.857769   1.462409e-19
    7            backup_online   31.209832   2.315902e-08
    8   proteccion_dispositivo   20.216007   6.917171e-06
    10            streaming_tv   17.320615   3.157429e-05
    11     streaming_peliculas   15.930611   6.570739e-05
    4         multiples_lineas    9.735960   1.806976e-03
    5            tipo_internet    9.715269   1.827433e-03
    0                   genero    0.254297   6.140655e-01
    3        servicio_telefono    0.092948   7.604618e-01
    //****************************************************//

 - Tabla de proporción por categoria

 ✅ Variables significativas (p < 0.05):
    Variable	Chi2 Score	p-value
    14	metodo_pago	466.119470	2.239792e-103
    12	tipo_contrato	392.706273	2.131687e-87
    6	seguridad_online	147.165601	7.219883e-34
    9	soporte_tecnico	135.439602	2.645952e-31
    2	posee_dependientes	131.271509	2.159540e-30
    13	factura_digital	104.979224	1.234232e-24
    1	posee_pareja	81.857769	1.462409e-19
    7	backup_online	31.209832	2.315902e-08
    8	proteccion_dispositivo	20.216007	6.917171e-06
    10	streaming_tv	17.320615	3.157429e-05
    11	streaming_peliculas	15.930611	6.570739e-05
    4	multiples_lineas	9.735960	1.806976e-03
    5	tipo_internet	9.715269	1.827433e-03

 
 ✅ Variables a Eliminar por evita multicolinealidad:
    Variable	Chi2 Score	p-value
    0	genero	0.254297	0.614066
    3	servicio_telefono	0.092948	0.760462

___________________________________________________________________________
🔍 **Resultados de Chi-Cuadrado (variables categoricas)**

🔹Variables más significativas (p-value extremadamente bajo)
Estas muestran una fuerte asociación con la cancelación y deben mantenerse:
- metodo_pago
- tipo_contrato
- seguridad_online
- soporte_tecnico
- posee_dependientes
- factura_digital
- posee_pareja

🔹 Variables con relevancia moderada
Aportan información, aunque con menor fuerza:
- backup_online
- proteccion_dispositivo
- streaming_tv
- streaming_peliculas
- multiples_lineas
- tipo_internet 

🔹 Variables con baja significancia
Son las candidatas a eliminar, ya que su asociación con Churn es débil:

- genero
- servicio_telefono 

**En resumen:** 
- Mantener: las variables con p-values muy bajos (aportan información clave). Aun que las variables multiples_lineas, tipo_internet son factibles a NO CONISDERAR.
- Eliminar o no considerar: genero y servicio_telefono.
- Esto permite reducir ruido en el modelo y concentrarse en las variables con mayor poder explicativo respecto a la cancelación de clientes (Churn).

_____________________________________________________________________________

## 🎯 Split -1 - Separando el Dataset ("X" e "y") 

- X (variables predictoras): contiene todas las demás características que explican o influyen en la cancelación.
- y (variable objetivo): contiene la información que queremos predecir (en este caso, si el cliente cancela o no).


## 🎯 Split -2 

- X se usa para entrenar el modelo.
- y se usa como etiqueta para supervisar el aprendizaje.
- Esto permite aplicar transformaciones solo a X (ej. escalado, encoding, imputación de valores).
- Compatibilidad con librerías de machine learning:
- Funciones como train_test_split, fit, predict en scikit-learn esperan que X y y estén separados.
- X debe ser una matriz de características, y y un vector con la variable objetivo.

🔍 **Resultados split-2 (X_train, X_test, y_train, y_test)**

🔹Datos para Entrenamiento

    - X_train : ´5.625´ Filas y ´16´ Columnas  (variables predictoras)

    - y_train : ´5.625 Filas y ´1´ Columna (Churn: variable objetivo)

🔹Datos para supervisión de aprendizaje (evaluación del modelo)

    - X_test  : ´1.407´ Filas y ´16´ Columnas (variables predictoras)

    - y_test  : ´1.407 Filas y ´1´ Columna (Churn: variable objetivo)


En esta etapa todas las varibles, NO se encuentran tranformadas para ejecución modelo (se aplicará en el siguente paso **One-Hot Encoding**)

________________________________________________________________________________

🔹 **¿Qué hace?**
- test_size=0.2 → el 20% de los datos se reserva para prueba, el 80% para entrenamiento.
- stratify=y → asegura que la proporción de clases (ej. clientes que cancelan vs. los que no) y se mantenga igual en ambos conjuntos. Esto es muy importante en problemas de Churn, donde suele haber desbalance.
- random_state=42 → fija la semilla aleatoria para que la división sea reproducible (al correr el código varias veces, obtendrás la misma separación).

________________________________________________________________________________

**En resumen:** 
- train_test_split → divide los datos en dos partes:
  - X_train, y_train: datos para entrenar el modelo.
  - X_test, y_test: datos para evaluar el modelo en ejemplos que nunca ha visto.

________________________________________________________________________________________________________________________

## 🎯 3. One-Hot Encoding

🔍 **Resultados One-Hot Encoding (X_train, X_test, y_train, y_test)**

🔹Datos para Entrenamiento

    - X_train : ´5.625´ Filas y ´21´ Columnas  (variables predictoras)

    - y_train : ´5.625 Filas y ´1´ Columna (Churn: variable objetivo)

🔹Datos para supevisar aprendizaje (evaluación del modelo)

    - X_test  : ´1.407´ Filas y ´21´ Columnas (variables predictoras)
    - y_test  : ´1.407 Filas y ´1´ Columna (Churn: variable objetivo)

En esta etapa todas las varibles, **se encuentran** tranformadas para ejecución de los modelos.

_________________________________________________________________________

🔹 **¿Qué hace?**

1. pd.get_dummies(X_train, drop_first=True)
      - Convierte las variables categóricas en variables binarias (dummy variables).
      - drop_first=True elimina la primera categoría para evitar multicolinealidad (redundancia).
      - Se aplica primero al conjunto de entrenamiento.
2. pd.get_dummies(X_test, drop_first=True)
      - Hace lo mismo en el conjunto de prueba.
      - El problema es que si el conjunto de prueba no contiene todas las categorías presentes en entrenamiento (o viceversa), las columnas resultantes pueden diferir.
3. X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
      - Alinea las columnas de ambos DataFrames.
      - Si alguna categoría existe en entrenamiento pero no en prueba, se crea la columna en prueba y se rellena con 0.
      - Así se asegura que entrenamiento y prueba tengan exactamente las mismas columnas, condición necesaria para que el modelo pueda hacer predicciones correctamente.

________________________________________________________________________________

**En resumen :** 

- Códifica variables categóricas en formato numérico (One-Hot Encoding).
- Evita multicolinealidad con drop_first=True.
- Garantiza consistencia entre entrenamiento y prueba, alineando las columnas y rellenando valores faltantes con 0.

 ________________________________________________________________________________________________________________________

🎯4. Verificar desbalanceamiento

    proportion
    Churn	
    0	73.422222
    1	26.577778
 ________________________________________________________________________________________________________________________


🎯5. Entrenamiento
🔹 Modelo 1 — Regresion logistica 🔹 Modelo 2 — Random Forest 
🔹 Modelo 3 — Regresion logistica con balanceo 🔹 Modelo 34— Regresion logistica con balanceo SMOTE


📊 Resumen ejecutivo de los resultados de los modelos predictivos


| Modelo                                | Clase | Precisión | Recall | F1-score | Accuracy | Macro avg | Weighted avg | Resumen |
|---------------------------------------|-------|-----------|--------|----------|----------|-----------|--------------|---------|
| **Regresión Logística (Normal)**      | No    | 0.84      | 0.90   | 0.87     | 0.80     | 0.72      | 0.79         | Detecta bien clientes que permanecen, pero pierde casi la mitad de los que cancelan. |
|                                       | Sí    | 0.65      | 0.52   | 0.58     | 0.80     | 0.72      | 0.79         | Moderado en churn, bajo recall. |
| **Random Forest (Normal)**            | No    | 0.83      | 0.90   | 0.87     | 0.79     | 0.71      | 0.78         | Similar a logística normal, buen desempeño en “No”. |
|                                       | Sí    | 0.64      | 0.50   | 0.56     | 0.79     | 0.71      | 0.78         | Moderado en churn, bajo recall. |
| **Regresión Logística (Balanceada)**  | No    | 0.91      | 0.73   | 0.81     | 0.75     | 0.72      | 0.76         | Alta precisión en “No”, pero sacrifica recall. |
|                                       | Sí    | 0.52      | 0.80   | 0.63     | 0.75     | 0.72      | 0.76         | Mejor recall en churn, aunque baja precisión. |
| **Regresión Logística (Balanceada + SMOTE)** | No | 0.88 | 0.79 | -        | 0.76     | 0.71–0.74 | 0.76–0.79     | Buen desempeño en “No”, algunos falsos positivos. |
|                                       | Sí    | 0.54      | 0.69   | -        | 0.76     | 0.71–0.74 | 0.76–0.79     | Mejora recall en churn, aunque con precisión moderada. |


________________________________________________________________________________________________________________________

🔍 Resultados de análisis Predictivo de Churn (Metodo aplicado en para procesamiento)

1. Preprocesamiento y Selección de Características:
  •	Se cargaron y procesaron los datos, incluyendo la eliminación de columnas irrelevantes ('id', 'cuentas_diarias', 'total_cobrado') debido a la multicolinealidad y baja significancia. Las variables 'genero' y 'servicio_telefono' fueron eliminadas por su débil asociación con el churn según el test Chi-cuadrado.
  •	Se aplicó One-Hot Encoding para convertir las variables categóricas a un formato numérico adecuado para los modelos.
  •	Se dividieron los datos en conjuntos de entrenamiento y prueba (80/20) y se escalaron las características numéricas.
2. Manejo del Desbalance de Clases:
  •	Se identificó un desbalance significativo en la variable objetivo 'Churn' (aproximadamente 73% 'No' y 27% 'Sí').
  •	Se aplicó la técnica SMOTE (Synthetic Minority Over-sampling Technique) para balancear el conjunto de entrenamiento, logrando una distribución del 50% para cada clase.
3. Evaluación de Modelos Predictivos:
  •	Se entrenaron dos modelos de clasificación:
    o	Regresión Logística (sin balanceo): Mostró una precisión general del 80%, pero un recall bajo para la clase 'Sí' (0.52), lo que significa que no detectaba a casi la mitad de los clientes que realmente cancelaban.
    o	Regresión Logística (con balanceo SMOTE): Mejoró el recall para la clase 'Sí' a 0.69, lo que indica una mayor capacidad para identificar a los clientes en riesgo de cancelación. La precisión para la clase 'Sí' fue de 0.54.
    o	Random Forest (con balanceo SMOTE): Ofreció un rendimiento similar al de la Regresión Logística balanceada, con un recall de 0.57 y precisión de 0.57 para la clase 'Sí'.

🎯 **Modelo Predictivo a Usar:**
Basándonos en los resultados obtenidos, el Modelo de Regresión Logística con balanceo de clases mediante SMOTE es el más recomendable para predecir la cancelación de clientes


________________________________________________________________________________________________________________________

##🎯Factores que más influyen en la cancelación

-	El estudio confirma que el churn (fuga de clientes) ocurre principalmente en etapas tempranas de la relación contractual, con una mediana de apenas 10 meses frente a 29 meses en clientes que permanecen.

-	Variables como tipo de internet (fibra óptica), uso de servicios de streaming y contratos mensuales incrementan el riesgo de abandono, mientras que la antigüedad del contrato y mayor gasto acumulado actúan como factores protectores.

-	Los modelos de regresión logística muestran que, sin balanceo, el desempeño es bueno para la clase mayoritaria (“No”), pero insuficiente para detectar la clase minoritaria (“Sí”).

-	Al aplicar SMOTE y class_weight balanced, el modelo mejora el recall de la clase “Sí” (0.69), logrando identificar más clientes en riesgo, aunque con menor precisión (0.54).

-	La matriz de confusión y los reportes de métricas confirman que el modelo balanceado ofrece un mejor equilibrio entre ambas clases, reduciendo falsos negativos en churn.


________________________________________________________________________________________________________________________

________________________________________________________________________________________________________________________

##🎯Conclusión 
Basándonos en los resultados obtenidos, el **Modelo de Regresión Logística con balanceo de clases mediante SMOTE** es el más recomendable para predecir la cancelación de clientes que ocurre en clientes de corta permanencia y con ciertos servicios asociados, ya que mejora la detección de clientes en riesgo, nos entrega un enfoque que maximiza la capacidad de retención y permite diseñar estrategias de retención más efectivas.


**Justificación:**

En un contexto de predicción de churn, es crucial maximizar la detección de los clientes que sí van a cancelar (recall de la clase 'Sí') para poder intervenir con estrategias de retención. Aunque el modelo de Regresión Logística balanceado mostró una precisión ligeramente menor para la clase 'Sí' en comparación con el modelo sin balanceo (0.54 vs 0.65), la mejora significativa en el recall (0.69 vs 0.52) es fundamental. Esto significa que el **modelo balanceado SMOTE** es capaz de identificar a más clientes que realmente abandonarán, lo cual es el objetivo principal de este tipo de predicción.



________________________________________________________________________________________________________________________

**FINALIZADO - Telecom X – Parte 2: Predicción de Cancelación (Churn)**
*/Desarrollado por Jaime Pradenas / Grupo G9 DataScience*
________________________________________________________________________________________________________________________
