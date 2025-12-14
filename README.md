# Calculadora de Riesgo – Cáncer de Endometrio (NEST)

> NSMP Endometrial Stratification Tool  
> Herramienta de apoyo a la decisión clínica (tipo “calculadora”) para estimar riesgo de recidiva y supervivencia en pacientes con cáncer de endometrio, con foco en el grupo NSMP.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Descripción del proyecto

Este proyecto nace en el reto “Hack the Uterus” del Hospital de la Santa Creu i Sant Pau. La idea es cubrir una necesidad muy concreta: dentro del cáncer de endometrio existe el grupo molecular NSMP (No Specific Molecular Profile), que representa aproximadamente la mitad de los casos y donde el pronóstico puede ser difícil de precisar con las guías actuales.

NEST intenta aportar una estimación más clara y fácil de contextualizar para el equipo clínico:

- Clasifica a la paciente en un grupo de riesgo (bajo / intermedio / alto).
- Estima probabilidades de DFS (supervivencia libre de enfermedad) y OS (supervivencia global) a 1, 2 y 3 años.
- Muestra de forma visual qué variables empujan el resultado hacia más o menos riesgo.

Nota: esto es un prototipo para soporte a la decisión clínica. No sustituye el juicio clínico ni las guías.

---

## Qué puede hacer la app

- Interfaz sencilla en Streamlit con un panel lateral para introducir variables clínicas, tumorales y moleculares.
- Asignación de grupo de riesgo (bajo, intermedio, alto) usando un enfoque de clustering.
- Estimación de supervivencia para DFS y OS a 1, 2 y 3 años.
- Explicabilidad (XAI): gráficos para entender qué variables han contribuido positiva o negativamente al riesgo.
- Comparación con la cohorte histórica para dar contexto al caso.
- Generación de un texto resumen listo para copiar en la historia clínica.

---

## Cómo funciona (resumen)

El enfoque es híbrido y combina dos piezas complementarias:

1) Clustering (K-Means)  
Se utiliza para encontrar patrones naturales en la cohorte histórica y agrupar pacientes en 3 perfiles. La separación se valida con Kaplan–Meier y log-rank test.

2) Regresión de Cox (supervivencia)  
Se entrena un modelo de Riesgos Proporcionales de Cox con variables clínicas en pacientes NSMP para generar un risk score individual.

Flujo de predicción:
1. El usuario introduce los datos clínicos.
2. El sistema imputa valores faltantes y estandariza los datos.
3. Se calcula el risk score con los coeficientes de Cox.
4. Se asigna el grupo de riesgo con umbrales predefinidos.

---

## Stack tecnológico

- Lenguaje: Python
- Interfaz: Streamlit
- Datos: Pandas, NumPy
- Machine Learning: Scikit-learn (preprocesado, KMeans), Scikit-survival (CoxPHSurvivalAnalysis)
- Visualización: Plotly

---

## Cómo ejecutar

1) Clonar el repositorio
```bash
git clone <URL_DEL_REPO>
cd <NOMBRE_DEL_REPO>
```

2) Instalar dependencias
```bash
pip install -r requirements.txt
```

3) Lanzar la app
```bash
streamlit run app_endometrio.py
```

---
## Estructura

├── app_endometrio.py        # Aplicación principal 
├── preprocessing.ipynb      # Limpieza e imputación 
├── clustering.ipynb         # Clustering + validación
├── bbdd_imputed_final.csv   # Dataset procesado 
├── requirements.txt         # Dependencias
└── README.md                # Documentación
