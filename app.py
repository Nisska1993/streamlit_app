import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Titre de l'application
st.title("Application de Régression Linéaire")
st.write("Cette application permet de réaliser une analyse de régression linéaire avec des données personnalisables.")

# Entrée des données
st.header("Entrée des Données")
st.write("Modifiez les données ci-dessous pour voir les résultats mis à jour.")

# Saisie des données
x_input = st.text_area("Entrez les valeurs de x (séparées par des virgules)", "1/1.192, 1/2.517, 1/4.571, 1/9.484")
y_input = st.text_area("Entrez les valeurs de y correspondantes (séparées par des virgules)", "3.76, 3.04, 2.76, 2.54")

# Transformation des données saisies
try:
    x = np.array([eval(i) for i in x_input.split(",")])
    y = np.array([float(i) for i in y_input.split(",")])

    if len(x) != len(y):
        st.error("Les longueurs de x et y doivent être identiques.")
    else:
        # Calcul manuel des paramètres
        n = np.size(x)
        x_mean, y_mean = np.mean(x), np.mean(y)

        Sxy = np.sum(x * y) - n * x_mean * y_mean
        Sxx = np.sum(x * x) - n * x_mean * x_mean

        b1 = Sxy / Sxx
        b0 = y_mean - b1 * x_mean
        y_pred_manual = b1 * x + b0

        # Erreur manuelle
        error = y - y_pred_manual
        se = np.sum(error**2)
        mse_manual = se / n
        rmse_manual = np.sqrt(mse_manual)
        SSt = np.sum((y - y_mean)**2)
        R2_manual = 1 - (se / SSt)

        # Régression avec Scikit-learn
        x = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        y_pred_sklearn = model.predict(x)

        mse_sklearn = mean_squared_error(y, y_pred_sklearn)
        rmse_sklearn = np.sqrt(mse_sklearn)
        r2_sklearn = r2_score(y, y_pred_sklearn)

        # Résultats de la régression
        st.header("Résultats de la Régression")
        st.write("### Calcul manuel")
        st.write(f"- Slope (b1): {b1}")
        st.write(f"- Intercept (b0): {b0}")
        st.write(f"- MSE: {mse_manual}")
        st.write(f"- RMSE: {rmse_manual}")
        st.write(f"- R²: {R2_manual}")

        st.write("### Scikit-learn")
        st.write(f"- Slope: {model.coef_[0]}")
        st.write(f"- Intercept: {model.intercept_}")
        st.write(f"- MSE: {mse_sklearn}")
        st.write(f"- RMSE: {rmse_sklearn}")
        st.write(f"- R²: {r2_sklearn}")

        # Visualisation
        st.header("Visualisation")
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='red', label='Données')
        ax.plot(x, y_pred_manual, color='green', label='Régression (manuel)')
        ax.plot(x, y_pred_sklearn, color='blue', linestyle='dashed', label='Régression (Scikit-learn)')
        ax.set_xlabel("1/pm")
        ax.set_ylabel("Permeability")
        ax.legend()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Erreur dans l'entrée des données : {e}")
