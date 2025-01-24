import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
import streamlit as st
import plotly.express as px

class StockAnalyzer:
    def __init__(self):
        self.API_KEY = "N8CI7VZN4G3SUO6M"
        self.symbol = "MSFT"
        self.df = None
        self.models = {}

    def fetch_data(self):
        """Obtener datos de Alpha Vantage"""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.symbol}&apikey={self.API_KEY}&datatype=csv"
        response = requests.get(url)
        
        if response.status_code == 200:
            # Guardar en CSV
            with open("stock_data.csv", "wb") as file:
                file.write(response.content)
            
            # Guardar en SQLite
            self.df = pd.read_csv("stock_data.csv")
            conn = sqlite3.connect('stock_data.db')
            self.df.to_sql('stock_prices', conn, if_exists='replace', index=False)
            conn.close()
            
            return True
        return False

    def preprocess_data(self):
        """Preprocesamiento de datos"""
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp")
        
        # Características adicionales
        self.df["volatility"] = (self.df["high"] - self.df["low"]) / self.df["low"] * 100
        self.df["daily_return"] = self.df["close"].pct_change() * 100
        self.df["rolling_avg_10"] = self.df["close"].rolling(window=10).mean()
        self.df["rolling_avg_30"] = self.df["close"].rolling(window=30).mean()
        self.df["relative_volume"] = self.df["volume"] / self.df["volume"].mean()
        
        # Eliminar valores nulos
        self.df = self.df.dropna()

    def perform_eda(self):
        """Análisis exploratorio de datos"""
        # Gráficos básicos
        self.plot_price_volume()
        self.plot_correlation_matrix()
        self.plot_seasonal_decomposition()
        
        # Análisis estadístico
        print("\nEstadísticas descriptivas:")
        print(self.df.describe())

    def train_models(self):
        """Entrenamiento de modelos"""
        X = self.df[["open", "high", "low", "volume"]]
        y = self.df["close"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(max_depth=3),
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor()
        }
        
        for name, model in models.items():
            # Entrenamiento
            model.fit(X_train, y_train)
            
            # Validación cruzada
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            
            # Predicciones
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nResultados para {name}:")
            print(f"RMSE (validación cruzada): {rmse_scores.mean():.2f} (+/- {rmse_scores.std() * 2:.2f})")
            print(f"R² Score: {r2:.2f}")
            
            self.models[name] = model

    def generate_trading_rules(self):
        """Sistema de reglas de trading"""
        tree_model = self.models['Decision Tree']
        rules = export_text(tree_model, feature_names=["open", "high", "low", "volume"])
        print("\nReglas de Trading generadas:")
        print(rules)
        return TradingRuleSystem(tree_model)

    def plot_price_volume(self):
        """Gráfico de precio y volumen"""
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(self.df["timestamp"], self.df["close"], color="blue")
        ax1.set_xlabel("Fecha")
        ax1.set_ylabel("Precio de cierre (USD)")
        
        ax2 = ax1.twinx()
        ax2.bar(self.df["timestamp"], self.df["volume"], alpha=0.3, color="orange")
        ax2.set_ylabel("Volumen")
        
        plt.title("Precio y Volumen a lo largo del tiempo")
        plt.show()

    def plot_correlation_matrix(self):
        """Matriz de correlación"""
        correlation_matrix = self.df[["open", "high", "low", "close", "volume"]].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Matriz de Correlación")
        plt.show()

    def plot_seasonal_decomposition(self):
        """Descomposición estacional"""
        decomposition = seasonal_decompose(self.df.set_index("timestamp")["close"], period=30)
        decomposition.plot()
        plt.show()

class TradingRuleSystem:
    def __init__(self, model):
        self.model = model
    
    def evaluate_conditions(self, data):
        prediction = self.model.predict([data])[0]
        current_price = data[0]
        
        if prediction > current_price * 1.02:
            return "COMPRAR"
        elif prediction < current_price * 0.98:
            return "VENDER"
        return "MANTENER"

def run_dashboard():
    st.title("Análisis de Acciones MSFT")
    
    analyzer = StockAnalyzer()
    if analyzer.fetch_data():
        analyzer.preprocess_data()
        
        # Mostrar gráficos
        st.subheader("Precio histórico")
        fig = px.line(analyzer.df, x="timestamp", y="close")
        st.plotly_chart(fig)
        
        # Predictor
        st.subheader("Predictor de precios")
        col1, col2 = st.columns(2)
        
        with col1:
            open_price = st.number_input("Precio de apertura", value=float(analyzer.df["open"].mean()))
            high_price = st.number_input("Precio más alto", value=float(analyzer.df["high"].mean()))
        
        with col2:
            low_price = st.number_input("Precio más bajo", value=float(analyzer.df["low"].mean()))
            volume = st.number_input("Volumen", value=float(analyzer.df["volume"].mean()))
        
        if st.button("Predecir"):
            analyzer.train_models()
            prediction = analyzer.models["Random Forest"].predict([[open_price, high_price, low_price, volume]])[0]
            st.success(f"Precio predicho: ${prediction:.2f}")

if __name__ == "__main__":
    # Para ejecutar el análisis completo
    analyzer = StockAnalyzer()
    analyzer.fetch_data()
    analyzer.preprocess_data()
    analyzer.perform_eda()
    analyzer.train_models()
    trading_system = analyzer.generate_trading_rules()
    
    # Para ejecutar el dashboard
    # run_dashboard()
