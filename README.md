# **Challenge3_TelecomX_LATAM Parte2- grupo G9**
**Desarrollado : Jaime Pradenas **
** Febrero 2026 **

- Pantilla Trello - https://trello.com/b/uH4K5Eek/telecomxlatamparte2jpraden
- Repositorio Github - https://github.com/jpraden/Challenge-DataScience-TelecomX-LATAM_Parte2

💡**Acerca del desafío**💡

**Descripción**

Telecom X – Parte 2: Predicción de Cancelación (Churn)

📣 Historia del Desafío

En la etapa anterior, se ha realizado el análisis exploratorio de la cancelación de clientes en Telecom X Parte1, bajo el rol "Asistente de análisis de datos", donde fue entregado con claridad los resultados y visión estratégica marcaron la diferencia.

Ahora, ¡he sido invitado oficialmente a formar parte del equipo de Machine Learning de la empresa!

🎯 Misión

Es desarrollar modelos predictivos capaces de prever qué clientes tienen mayor probabilidad de cancelar sus servicios.

La empresa quiere anticiparse al problema de la cancelación, y en este rol corresponde construir un pipeline robusto para esta etapa inicial de modelado.

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
- Google Colab - Entorno de desarrollo (Python)
  * Pandas – Manipulación y análisis de datos
  * Matplotlib – Visualización de datos
  * Seaborn – Visualizaciones estadísticas
  * NumPy – Operaciones numéricas

📦 Instalación y Configuración
- Prerrequisitos Python 3.8 o superior
  * pandas>=1.5.0
  * matplotlib>=3.5.0
  * seaborn>=0.12.0
  * numpy>=1.23.0

___________________________________________________________________________________________________________


🛠️ Repositorio del proyecto
- Github --> https://github.com/jpraden/Challenge-DataScience-TelecomX-LATAM_Parte2/TelecomX_LATAM.ipynb

🛠️ Estructura Carpetas en GitHub

../Challenge-DataScience-TelecomX-LATAM_Parte2

    -> TelecomX_LATAM_Parte2
    -> README.md

     ../datos
       -> datos_tratados.csv
 

___________________________________________________________________________________________________________-

🚀 Ejecución del proyecto
- Opción 1: 
  * Abrir cuaderno desde Google Colab (Recomendado)
  * Ejecutar Todo o secuencialmente de nueva casilla

- Opción 2: Jupyter notebook
- Opción 3: Visual Studio Code


___________________________________________________________________________________________________________

📊 #**Actividades realizadas**


#Objetivo:

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

#📌🔧 Eliminación de Columnas Irrelevantes

**Descripción:**

Eliminando columnas que no aportan valor al análisis o a los modelos predictivos, como identificadores únicos (por ejemplo, el ID del cliente). Estas columnas no ayudan en la predicción de la cancelación y pueden incluso perjudicar el desempeño de los modelos.

_____________________________________________________________________________
_____________________________________________________________________________


#📊 Insights claves


#📊Recomendaciones para el equipo de Data Science


______________________________________________________________________________

#📊Conclusión final

______________________________________________________________________________

**FINALIZADO / Challenge2_TelecomX_LATAM - Análisis de Evasión de Clientes / Desarrollado por Jaime Pradenas / Grupo G9 DataScience**

______________________________________________________________________________
