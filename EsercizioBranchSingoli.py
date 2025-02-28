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


# Branch 3 - Visualizzazioni grafiche
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['specie'], y=df['sepal length (cm)'])
plt.title("Distribuzione della lunghezza del sepalo per specie")
plt.xlabel("Specie")
plt.ylabel("Lunghezza del sepalo (cm)")
plt.show()
print("Il boxplot mostra la distribuzione della lunghezza del sepalo per ogni specie. Si possono vedere differenze tra le varie specie.")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['sepal length (cm)'], y=df['sepal width (cm)'], hue=df['specie'], palette='viridis')
plt.title("Relazione tra lunghezza e larghezza del sepalo")
plt.xlabel("Lunghezza del sepalo (cm)")
plt.ylabel("Larghezza del sepalo (cm)")
plt.show()
print("Lo scatter plot mostra la relazione tra lunghezza e larghezza del sepalo per ogni specie. Si possono notare raggruppamenti e possibili correlazioni.")