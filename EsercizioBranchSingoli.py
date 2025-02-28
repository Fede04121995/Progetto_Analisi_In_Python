import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#Esplorazione iniziale con importazione del dataset e visualizzazione dello stesso
iris = load_iris()
X = iris.data
y = iris.target 

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['specie'] = y

df.head()
df.describe()
df.info()

#Informazioni di base come dimenzione, tipo di dato ed eventuali valori mancanti
print("Dimensioni del dataset:", df.shape)
print("\nTipi di dati:\n", df.dtypes)
print("\nValori mancanti:\n", df.isnull().sum())

# Conteggio delle specie
print("\nConteggio delle specie:")
print(df['specie'].value_counts())

# Branch 2 - Analisi approfondita
print("\nMatrice di correlazione")
print(df.corr())  

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matrice di Correlazione delle Variabili")
plt.show()

# Raggruppamenti per specie
print("\n--- Raggruppamenti per specie ---")
gruppo_specie = df.groupby('specie')
print(gruppo_specie.size())

# Confronto tra media e massimo per ogni caratteristica e specie
print("\n--- Confronto tra valori medi e massimi per ogni caratteristica ---")
confronto = gruppo_specie.agg(['mean', 'max'])
print(confronto)


