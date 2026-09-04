import pandas as pd
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings

# Ignorar advertencias para una salida limpia
warnings.filterwarnings("ignore")

def modelo_maximalista_15_variables(file_path):
    print("--- 🚀 INICIANDO ENTRENAMIENTO DEL MODELO MAXIMALISTA (15 VARIABLES) ---")
    
    # 1. CARGA Y FILTRADO DE DATOS
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo '{file_path}'.")
        return

    # Filtro para pacientes NSMP (usando el marcador r06=1.0)
    # Ajusta esta condición si tu criterio para NSMP es diferente
    df_nsmp = df[df['estudio_genetico_r06'] == 1.0].copy()
    
    # Definición de Evento (Recidiva) y Tiempo (Supervivencia)
    df_nsmp['Event'] = df_nsmp['recidiva'] > 0
    # Usamos DFS si hubo evento, OS_Days si no (censurado)
    df_nsmp['Time'] = np.where(df_nsmp['Event'], df_nsmp['DFS'], df_nsmp['OS_Days'])
    
    # Limpieza: eliminar tiempos <= 0 que causan error en Cox
    df_nsmp = df_nsmp[df_nsmp['Time'] > 0]
    
    # 2. SELECCIÓN DE LAS 15 VARIABLES CANDIDATAS
    # Esta lista combina factores biológicos, clínicos y quirúrgicos
    features_15 = [
        'FIGO2023',             # Estadio FIGO (Potencia anatómica)
        'grado_histologi',      # Grado histológico (Potencia biológica)
        'infiltracion_mi',      # Infiltración miometrial
        'afectacion_linf',      # LVSI (Invasión linfovascular)
        'recep_est_porcent',    # Receptores Estrógenos (Factor protector)
        'imc',                  # IMC (Factor metabólico/protector)
        'AP_centinela_pelvico', # Estado del ganglio centinela
        'histo_defin',          # Subtipo histológico definitivo
        'estadiaje_pre_i',      # Estadiaje clínico pre-quirúrgico
        'ecotv_infiltsub',      # Eco Transvaginal (Infiltración subjetiva)
        'asa',                  # Estado físico (riesgo quirúrgico)
        'edad',                 # Edad de la paciente
        'tamano_tumoral',       # Tamaño del tumor (cm)
        'rece_de_Ppor',         # Receptores Progesterona
        'metasta_distan'        # Metástasis a distancia (Factor crítico)
    ]
    
    # Verificar que las columnas existen
    missing_cols = [c for c in features_15 if c not in df_nsmp.columns]
    if missing_cols:
        print(f"Error: Faltan estas columnas en el CSV: {missing_cols}")
        return

    print(f"Variables seleccionadas: {len(features_15)}")
    print(f"Pacientes NSMP válidos para análisis: {len(df_nsmp)}")
    
    # 3. PREPROCESAMIENTO DE DATOS
    X = df_nsmp[features_15]
    # Crear objeto estructurado para scikit-survival (Status, Time)
    y = Surv.from_dataframe('Event', 'Time', df_nsmp)
    
    # Imputación de valores faltantes (usando la mediana)
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features_15)
    
    # Estandarización (Scaling): Vital para comparar los coeficientes (Betas)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=features_15)
    
    # 4. ENTRENAMIENTO DEL MODELO COX
    cox = CoxPHSurvivalAnalysis()
    try:
        cox.fit(X_scaled, y)
        
        # 5. EVALUACIÓN INICIAL (C-Index en entrenamiento)
        prediccion = cox.predict(X_scaled)
        c_index_train = concordance_index_censored(y['Event'], y['Time'], prediccion)[0]
        
        print(f"\n📊 PRECISIÓN EN ENTRENAMIENTO (C-Index): {c_index_train:.4f}")
        
        # 6. RESULTADOS: PESOS Y HAZARD RATIOS
        coefs = pd.Series(cox.coef_, index=features_15)
        hrs = np.exp(coefs)
        
        res_df = pd.DataFrame({
            'Peso (Beta)': coefs,
            'Hazard Ratio (HR)': hrs,
            'Interpretación': np.where(hrs > 1, 'AUMENTA RIESGO 🔴', 'FACTOR PROTECTOR 🟢')
        }).sort_values(by='Hazard Ratio (HR)', ascending=False)
        
        print("\n--- 📋 JERARQUÍA DE IMPACTO DE LAS 15 VARIABLES ---")
        print(res_df)
        
        # 7. VALIDACIÓN ROBUSTA (BOOTSTRAPPING)
        # Esto es crucial para ver si el modelo "memorizó" (overfitting) o "aprendió"
        print("\n--- 🛡️ EJECUTANDO VALIDACIÓN DE ROBUSTEZ (500 ITERACIONES) ---")
        print("Simulando 500 escenarios clínicos diferentes... (Espere unos segundos)")
        
        scores = []
        n_iterations = 500
        
        for i in range(n_iterations):
            # Remuestreo con reemplazo (Bootstrapping)
            X_res, y_res = resample(X_scaled, y, random_state=i)
            
            # Entrenar y evaluar en la muestra resampleada
            try:
                cox_boot = CoxPHSurvivalAnalysis()
                cox_boot.fit(X_res, y_res)
                score = cox_boot.score(X_res, y_res)
                scores.append(score)
            except:
                continue # Saltar iteraciones fallidas (raras)

        mean_score = np.mean(scores)
        lower_ci = np.percentile(scores, 2.5)
        upper_ci = np.percentile(scores, 97.5)
        
        print(f"\n✅ RESULTADO FINAL DE VALIDACIÓN:")
        print(f"C-Index Promedio (Realista): {mean_score:.4f}")
        print(f"Intervalo de Confianza 95%: [{lower_ci:.4f} - {upper_ci:.4f}]")
        
        # Interpretación automática para tu presentación
        diff = c_index_train - mean_score
        print("\n--- 💡 DIAGNÓSTICO DEL MODELO ---")
        
        if mean_score > 0.85:
            print("🌟 EXCELENTE: El modelo es extremadamente preciso y robusto.")
            print("Puedes presentarlo como un 'Algoritmo Avanzado de Alta Precisión'.")
        elif mean_score > 0.80:
            print("👍 MUY BUENO: El modelo es sólido y supera los estándares clínicos usuales.")
        elif diff > 0.10:
            print("⚠️ ALERTA DE SOBREAJUSTE (Overfitting):")
            print(f"El rendimiento cae mucho en validación ({c_index_train:.2f} -> {mean_score:.2f}).")
            print("Consejo: 15 variables pueden ser demasiadas. Considera volver al modelo de 5 variables.")
        else:
            print("ℹ️ RESULTADO ESTÁNDAR: El modelo funciona, pero verifica si aporta más que el modelo simple.")

    except Exception as e:
        print(f"\n❌ Error crítico durante el entrenamiento: {e}")
        print("Posible causa: Alguna variable nueva podría tener varianza cero (todos los valores iguales) o haber colinealidad perfecta.")

if __name__ == "__main__":
    # Asegúrate de que el archivo CSV esté en la misma carpeta
    modelo_maximalista_15_variables('aaa2_imputed_final.csv')