# -*- coding: utf-8 -*-
"""
Predicción Meteorológica: Análisis de Patrones de Lluvia con Técnicas de Minería de Datos
Versión mejorada con análisis detallado y visualizaciones avanzadas
"""

# Importación de bibliotecas con manejo de errores
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime
    
    # Configuración de estilo para visualizaciones
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12
except ImportError as e:
    print(f"Error importando bibliotecas: {e}")

def cargar_datos(ruta_archivo):
    """Carga y muestra información inicial del dataset"""
    try:
        df = pd.read_parquet(ruta_archivo)
        print(f"Dataset cargado exitosamente. Dimensiones: {df.shape}")
        print("\nInformación del dataset:")
        print(df.info())
        print("\nPrimeras 5 filas:")
        display(df.head())
        return df
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

def analizar_valores_faltantes(df):
    """Análisis detallado de valores faltantes con visualización"""
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores Faltantes': missing_values,
        'Porcentaje': missing_percent.round(2)
    }).sort_values('Porcentaje', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_df.index, y='Porcentaje', data=missing_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Porcentaje de Valores Faltantes por Columna')
    plt.tight_layout()
    plt.show()
    
    return missing_df

def analisis_estadistico(df, columnas_numericas):
    """Análisis estadístico detallado con visualizaciones"""
    stats = df[columnas_numericas].describe()
    
    fig, axes = plt.subplots(len(columnas_numericas), 2, 
                            figsize=(15, 5*len(columnas_numericas)))
    
    for i, col in enumerate(columnas_numericas):
        # Histograma con KDE
        sns.histplot(data=df, x=col, ax=axes[i,0], kde=True)
        axes[i,0].set_title(f'Distribución de {col}')
        
        # Box plot
        sns.boxplot(data=df, y=col, ax=axes[i,1])
        axes[i,1].set_title(f'Box Plot de {col}')
    
    plt.tight_layout()
    plt.show()
    
    return stats

def limpiar_dataset(df):
    """Limpieza y preparación avanzada del dataset"""
    df_limpio = df.copy()
    
    # Tratamiento de valores numéricos
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
    
    # Tratamiento de valores categóricos
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0])
    
    # Normalización de variables numéricas
    scaler = StandardScaler()
    df_limpio[num_cols] = scaler.fit_transform(df_limpio[num_cols])
    
    return df_limpio

def analisis_temporal(df):
    """Análisis temporal detallado"""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Mes'] = df['Date'].dt.month
    df['Año'] = df['Date'].dt.year
    
    # Tendencia temporal de temperaturas
    plt.figure(figsize=(15, 6))
    df.groupby('Mes')[['MinTemp', 'MaxTemp']].mean().plot(kind='line', marker='o')
    plt.title('Tendencia de Temperaturas por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Temperatura')
    plt.grid(True)
    plt.show()
    
    # Análisis por año
    plt.figure(figsize=(15, 6))
    df.groupby('Año')['RISK_MM'].mean().plot(kind='bar')
    plt.title('Promedio de Riesgo de Lluvia por Año')
    plt.xlabel('Año')
    plt.ylabel('RISK_MM Promedio')
    plt.tight_layout()
    plt.show()

def analisis_correlaciones(df):
    """Análisis de correlaciones mejorado"""
    num_cols = df.select_dtypes(include=['float64']).columns
    corr_matrix = df[num_cols].corr()
    
    plt.figure(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Matriz de Correlaciones')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def analisis_completo():
    """Función principal que ejecuta todo el análisis"""
    # Cargar datos
    df = cargar_datos('weatherAUS.parquet')
    if df is None:
        return
    
    # Análisis de valores faltantes
    print("\nAnálisis de valores faltantes:")
    missing_analysis = analizar_valores_faltantes(df)
    display(missing_analysis)
    
    # Análisis estadístico
    print("\nAnálisis estadístico:")
    columnas_numericas = ['MinTemp', 'MaxTemp', 'RISK_MM']
    stats = analisis_estadistico(df, columnas_numericas)
    display(stats)
    
    # Limpieza de datos
    print("\nLimpieza de datos:")
    df_limpio = limpiar_dataset(df)
    
    # Análisis temporal
    print("\nAnálisis temporal:")
    analisis_temporal(df_limpio)
    
    # Análisis de correlaciones
    print("\nAnálisis de correlaciones:")
    correlaciones = analisis_correlaciones(df_limpio)
    display(correlaciones)

if __name__ == "__main__":
    analisis_completo()