# NEST — Endometrial Cancer Risk Calculator

> **NSMP Endometrial Stratification Tool**  
> Clinical decision-support prototype for estimating recurrence risk and survival in NSMP endometrial cancer patients.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## Overview

Developed during the **"Hack the Uterus"** hackathon at Hospital de la Santa Creu i Sant Pau. NEST addresses a specific clinical need: within endometrial cancer, the **NSMP** (No Specific Molecular Profile) group represents ~50% of cases where prognosis is difficult to assess with current guidelines.

The tool provides clinicians with:

- **Risk group classification** (low / intermediate / high)
- **Survival estimates** for DFS (disease-free survival) and OS (overall survival) at 1, 2, and 3 years
- **Explainability (XAI)** — visual breakdown of which variables push risk up or down
- **Clinical summary text** ready to copy into patient records

> **Disclaimer:** This is a research prototype for clinical decision support. It does not replace clinical judgment or established guidelines.

## How It Works

Hybrid approach combining two complementary methods:

1. **K-Means clustering** — identifies natural patient profiles in the historical cohort; separation validated with Kaplan–Meier curves and log-rank tests
2. **Cox proportional hazards regression** — trains on NSMP patients to generate an individual risk score

**Prediction flow:**
1. Clinician enters patient data via the Streamlit sidebar
2. System imputes missing values and standardizes inputs
3. Cox coefficients produce a risk score
4. Predefined thresholds assign a risk group

## Tech Stack

- Python, Streamlit
- Pandas, NumPy
- scikit-learn (preprocessing, KMeans), scikit-survival (CoxPH)
- Plotly (interactive visualizations)

## Project Structure

```
Bits-Per-la-Marato/
├── app.py                          # Main Streamlit application
├── src/                            # Python modules
│   └── survival_model.py           # Survival model logic
├── notebooks/                      # Exploratory & preprocessing notebooks
│   ├── preprocessing.ipynb
│   └── clustering.ipynb
├── data/                           # Datasets
│   ├── raw/
│   │   └── IQ_Cancer_Endometrio_merged_NMSP.xlsx
│   └── processed/
│       └── bbdd_imputed_final.csv
├── outputs/                        # Cluster analysis results & figures
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Paula Esteve Sabater

## Context

Hackathon project — Hack the Uterus, Hospital de la Santa Creu i Sant Pau.
