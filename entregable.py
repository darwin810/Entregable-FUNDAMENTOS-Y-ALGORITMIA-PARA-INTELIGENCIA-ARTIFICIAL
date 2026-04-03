import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

wine = load_wine()

# lo pasamos a dataframe para trabajarlo mejor con pandas
frame = pd.DataFrame(data=wine.data, columns=wine.feature_names)
frame['etiqueta'] = wine.target
frame['nombre_clase'] = wine.target_names[wine.target]

print(frame.head())
print(frame.shape)
print(wine.target_names)

#procesamiento con (pandas)
print(frame.describe())
print(frame.isnull().sum())
print(frame['nombre_clase'].value_counts())
print(frame.groupby('nombre_clase')['alcohol'].mean())
print(frame.groupby('nombre_clase')['flavanoids'].mean())

"""GRafico con groupby"""

reporte1 = frame.groupby('nombre_clase')['alcohol'].mean()
reporte2 = frame.groupby('nombre_clase')['nombre_clase'].count()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('analisis por clase', fontsize=13)
colores = ['b', 'r', 'c']
axes[0].bar(reporte1.index, reporte1.values, color=colores, edgecolor='black')
axes[0].set_title('promedio de alcohol por clase')
axes[0].set_xlabel('clase')
axes[0].set_ylabel('alcohol (%)')
axes[0].grid(axis='y', alpha=0.4)
for i, v in enumerate(reporte1.values):
    axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center')

axes[1].pie(reporte2.values, labels=reporte2.index, autopct='%1.1f%%',
            colors=colores)
axes[1].set_title('distribucion de clases')

plt.tight_layout()

#procesamiento de daatos

x = wine.data
y = wine.target

frame2 = frame.drop(columns=['etiqueta', 'nombre_clase'])
nulos = frame2.isnull().sum().sum()
print(f'valores nulos: {nulos}')
scaler = StandardScaler()
x_escalado = scaler.fit_transform(x)
print('antes de escalar:', x[:3, 0])
print('despues de escalar:', x_escalado[:3, 0])
print('media:', x_escalado[:, 0].mean())
print('std:', x_escalado[:, 0].std())

for i, nombre in enumerate(wine.target_names):
    print(f'{nombre} -> {i}')

x_train, x_test, y_train, y_test = train_test_split(x_escalado, y, test_size=0.2, random_state=42)

print('entrenamiento:', x_train.shape)
print('prueba:', x_test.shape)

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

predicciones = model.predict(x_test)
print(classification_report(y_test, predicciones))
print(confusion_matrix(y_test, predicciones))

print('etiquetas reales:')
print(y_test)
print('predicciones del modelo:')
print(predicciones)
print(classification_report(y_test, predicciones, target_names=wine.target_names))
print(confusion_matrix(y_test, predicciones))

muestra = x_test[0].reshape(1, -1)
pred = model.predict(muestra)
print(f'clase predicha: {wine.target_names[pred[0]]}')
print(f'clase real: {wine.target_names[y_test[0]]}')

#====analisis estadistico=
datos = np.array([15, 21, 18, 14, 20, 35, 19, 22, 17, 16, 45, 13, 23, 18, 21])
print('datos:', datos)

media = np.mean(datos)
mediana = np.median(datos)
print(f'media: {media}')
print(f'mediana: {mediana}')

varianza = np.var(datos)
std = np.std(datos)
print(f'varianza: {varianza}')
print(f'desviacion estandar: {std}')

limite_inf = media - 2 * std
limite_sup = media + 2 * std
atipicos = datos[np.abs(datos - media) > 2 * std]
normales = datos[np.abs(datos - media) <= 2 * std]
print(f'limite inferior: {limite_inf:.2f}')
print(f'limite superior: {limite_sup:.2f}')
print(f'valores normales: {normales}')
print(f'valores atipicos: {atipicos}')



#grafico(hisograma)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('analisis estadistico', fontsize=13)

axes2[0].hist(datos, bins=8, color='c', edgecolor='black', alpha=0.8)
axes2[0].axvline(media, color='red', linestyle='--', label=f'media={media:.2f}')
axes2[0].axvline(mediana, color='green', linestyle='--', label=f'mediana={mediana:.2f}')
axes2[0].axvline(limite_inf, color='orange', linestyle=':', label=f'lim inf={limite_inf:.2f}')
axes2[0].axvline(limite_sup, color='orange', linestyle=':', label=f'lim sup={limite_sup:.2f}')
axes2[0].set_title('histograma')
axes2[0].set_xlabel('valor')
axes2[0].set_ylabel('frecuencia')
axes2[0].legend(fontsize=8)
axes2[0].grid(alpha=0.3)
axes2[1].boxplot(datos, vert=True, patch_artist=True,
                 boxprops=dict(facecolor='b', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2))
axes2[1].set_title('boxplot - valores atipicos')
axes2[1].set_ylabel('valor')
axes2[1].grid(alpha=0.3)
for val in atipicos:
    axes2[1].plot(1, val, 'ro', markersize=10, label=f'atipico: {val}')
axes2[1].legend()
plt.tight_layout()
plt.show()

