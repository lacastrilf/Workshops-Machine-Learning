# Workshop 3: Segmentación de clientes con Clustering (K-Means vs. Agglomerative vs. DBSCAN)

## 🎯 Objetivo

¡Hola a todos! En este workshop vamos a retomar lo que ya trabajamos con el dataset de Iris, pero esta vez con un problema **real de negocio**: segmentar a los clientes de un centro comercial según su comportamiento de compra.

Van a aplicar y comparar **tres estrategias de clustering no supervisado**:

- **K-Means**
- **Clustering jerárquico aglomerativo**
- **DBSCAN**

Y a partir de ese análisis comparativo, su misión es responder una pregunta de negocio muy concreta:

> **¿En cuántos grupos "naturales" se dividen los clientes de este centro comercial, y qué caracteriza a cada uno?**

Al final, van a tomar esos grupos (que ustedes mismos descubrieron) y los van a convertir en una variable objetivo para entrenar un clasificador. Es decir: pasamos de **no supervisado → supervisado**, usando lo primero para construir lo segundo. Este es un patrón MUY común en la industria (clustering para generar etiquetas cuando no existen, y luego clasificación para poder predecir el segmento de un cliente nuevo sin tener que re-ejecutar el clustering cada vez).

No se preocupen si no dominan cada método al 100% — en cada sección van a encontrar pistas de código, estructuras para completar, y preguntas que los van a guiar paso a paso. La idea es que trabajen con guía, no que memoricen sintaxis.

---

## ⏱️ Duración y ritmo sugerido (90 min)

Este es un workshop de clase, así que el tiempo importa. Esta es una distribución sugerida — ajústenla según el ritmo de su equipo, pero no se queden pegados demasiado tiempo en una sola fase:

| Fase | Contenido | Tiempo sugerido |
|---|---|---:|
| 0 | Carga y preprocesamiento | 10 min |
| 1 | K-Means | 15 min |
| 2 | Agglomerative Clustering | 10 min |
| 3 | DBSCAN | 15 min |
| 4 | Visualización PCA 3D | 10 min |
| 5 | Comparación con Silhouette Score | 5 min |
| 6 | Definir las clases naturales | 5 min |
| 7 | Interpretación de los clusters (perfilado) | 10 min |
| 8-10 | Etiquetado + clasificador + evaluación | 10 min |

---

## 1. El dataset

Vamos a trabajar con **Mall Customer Segmentation Data**, disponible en Kaggle:

🔗 https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

El archivo `Mall_Customers.csv` contiene 200 clientes de un centro comercial, con las siguientes columnas:

| Columna | Descripción |
|---|---|
| `CustomerID` | Identificador único del cliente (no es una característica, es un índice) |
| `Genre` | Género del cliente (**ojo**: la columna se llama literalmente `Genre`, no `Gender` — es un detalle del dataset original, no un error suyo) |
| `Age` | Edad del cliente |
| `Annual Income (k$)` | Ingreso anual en miles de dólares |
| `Spending Score (1-100)` | Puntaje de gasto asignado por el centro comercial según comportamiento de compra (1 = gasta poco, 100 = gasta mucho) |

Pueden descargarlo de Kaggle y cargarlo localmente, o usar `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
print(path)
```

---

## 2. Fase 0 — Carga y preprocesamiento de datos

Antes de lanzarse a clusterizar, como siempre, hay que conocer y preparar los datos.

### 2.1. Exploración inicial

```python
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")

df.head()
df.shape
df.isnull().sum()
```
**Añadan otros pasos de exploracion que consideren relevantes**

### Pregunta

- ¿Hay valores nulos?
- ¿Qué tipo de dato tiene cada columna? ¿Cuáles son numéricas y cuál es categórica?

### 2.2. Selección de variables

`CustomerID` no aporta información para el clustering — es solo un identificador. Elimínenla.

Para esta segmentación, vamos a usar **las 4 variables restantes**: `Genre`, `Age`, `Annual Income (k$)` y `Spending Score (1-100)`.

### 2.3. Codificación de la variable categórica

`Genre` es categórica (Male/Female), y los algoritmos de clustering que vamos a usar trabajan con distancias numéricas. Necesitan codificarla.

> **Pista:** piensen en lo que ya vieron de encoding. ¿`Genre` tiene un orden natural? ¿Cuántas categorías tiene? Esto les dice qué tipo de encoding es el apropiado aquí (recuerden `LabelEncoder`, `OneHotEncoder` o incluso mapear manualmente con `.map({...})` dado que solo son 2 categorías).

```python
from sklearn.preprocessing import ...

# Completen: codifiquen la columna Genre

```

### 2.4. Estandarización

```python
from sklearn.preprocessing import StandardScaler

features = ["Genre_encoded", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
```

### Pregunta

- ¿Por qué es importante estandarizar antes de aplicar K-Means, Agglomerative Clustering o DBSCAN? (Piensen específicamente en qué pasaría si NO estandarizaran — comparen las escalas de `Age` vs. `Annual Income (k$)`).

---

## 3. Fase 1 — K-Means

K-Means es el método que probablemente no vieron en el ejercicio de Iris, así que aquí va un repaso rápido: K-Means necesita que le digan de antemano **cuántos clusters (k)** quiere formar, y va iterando hasta encontrar k centroides que minimizan la distancia de cada punto a su centroide más cercano.

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=k,     # <- completen
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X_scaled)
```

### 3.1. Método del codo (Elbow Method)

Para elegir un buen `k`, una técnica clásica es graficar la **inercia** (`model.inertia_`, la suma de distancias al cuadrado de cada punto a su centroide) contra distintos valores de k, y buscar el "codo" de la curva — el punto donde agregar más clusters deja de reducir mucho la inercia.

Prueben, como mínimo:

```text
k = 2, 3, 4, 5, 6, 7, 8
```

> **Pista de estructura:**
> ```python
> inertias = []
> ks = range(2, 9)
>
> for k in ks:
>     model = KMeans(n_clusters=k, random_state=42, n_init=10)
>     model.fit(X_scaled)
>     inertias.append(model.inertia_)
>
> # Ahora grafiquen ks vs. inertias con matplotlib
> ```

### 3.2. Silhouette Score para cada k

Igual que hicieron con Iris, calculen el Silhouette Score para cada valor de `k` y construyan su tabla comparativa:

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)
```

| k | Inercia | Silhouette Score |
|---:|---:|---:|
| 2 | ... | ... |
| 3 | ... | ... |
| ... | ... | ... |

### Preguntas

1. ¿El codo de la curva de inercia coincide con el `k` de mayor Silhouette Score?
2. ¿Qué `k` seleccionarían para K-Means? Justifiquen.

---

## 4. Fase 2 — Clustering jerárquico aglomerativo

Repitan aquí el mismo flujo que ya conocen del ejercicio de Iris.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method="ward")
dendrogram(Z)
```

Prueben los mismos valores de `k = 2, 3, 4, 5, 6, 7, 8` con `linkage="ward"`, calculen el Silhouette Score para cada uno, y construyan su tabla comparativa (igual que hicieron antes).

### Preguntas

- ¿Cuántos grupos sugiere el dendrograma visualmente?
- ¿Coincide con el mejor `k` según Silhouette Score?

---

## 5. Fase 3 — DBSCAN

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=...,          # <- exploren varios valores
    min_samples=...   # <- exploren varios valores
)

labels = model.fit_predict(X_scaled)
```

> **Recuerden:** DBSCAN no necesita que le digan `k` de antemano — el número de clusters emerge de los parámetros `eps` y `min_samples`. 

Exploren combinaciones razonables, por ejemplo:

```text
eps = 0.3, 0.5, 0.7, 0.9, 1.1
min_samples = 3, 5, 8, 10
```

No necesitan probar todas las combinaciones — en cuanto encuentren una región de parámetros que dé un resultado razonable (ni 1 solo cluster gigante, ni 190 puntos marcados como ruido), pueden quedarse con esa.

Para el Silhouette Score, recuerden excluir el ruido:

```python
mask = labels != -1
score = silhouette_score(X_scaled[mask], labels[mask])
```

### Preguntas

1. ¿Cuántos clusters encontró DBSCAN con los parámetros que eligieron?
2. ¿Cuántos puntos quedaron como ruido?
3. ¿Qué pasa con el número de clusters si aumentan `eps`? ¿Y si aumentan `min_samples`?

---

## 6. Fase 4 — Visualización con PCA en 3D

Como estamos usando 4 variables (`Genre_encoded`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`), no podemos graficar directamente en 2D o 3D sin antes reducir la dimensionalidad. Vamos a usar **PCA a 3 componentes** para poder visualizar los clusters de cada método en un mismo espacio comparable.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
print("Varianza explicada acumulada:", pca.explained_variance_ratio_.sum())
```

### Pregunta

- ¿Qué porcentaje de la varianza total logran capturar las 3 componentes? ¿Les parece suficiente para confiar en la visualización?

### Estructura para el gráfico 3D

Deben generar **un gráfico 3D por cada método de clustering** (K-Means, Agglomerative, DBSCAN), coloreando los puntos según la etiqueta de cluster asignada por cada uno. Aquí tienen el esqueleto — solo deben completar las etiquetas (`labels`) de cada método:

```python
import matplotlib.pyplot as plt

def plot_clusters_3d(X_pca, labels, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
        c=labels,
        cmap="viridis",
        s=40
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)

    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)

    plt.show()

# Ejemplo de uso (repitan para cada método):
# plot_clusters_3d(X_pca, labels_kmeans, "K-Means")
# plot_clusters_3d(X_pca, labels_agglomerative, "Agglomerative Clustering")
# plot_clusters_3d(X_pca, labels_dbscan, "DBSCAN")
```

> **Tip:** para DBSCAN, los puntos de ruido (`-1`) también se van a colorear — fíjense si eso dificulta la lectura del gráfico, y coméntenlo en su análisis.

---

## 7. Fase 5 — Comparación final con Silhouette Score

Construyan una tabla que resuma los tres métodos, con la mejor configuración que encontraron para cada uno:

| Método | Parámetros | N° de clusters | N° de puntos de ruido | Silhouette Score |
|---|---|---:|---:|---:|
| K-Means | `k = ...` | ... | N/A | ... |
| Agglomerative | `k = ..., linkage="ward"` | ... | N/A | ... |
| DBSCAN | `eps = ..., min_samples = ...` | ... | ... | ... |

---

## 8. Fase 6 — ¿Cuáles son las clases naturales?

Con la tabla anterior en la mano, respondan la pregunta central del workshop:

> **Basándose en su análisis comparativo (Silhouette Score, número de clusters, cantidad de ruido, y lo que observaron en las visualizaciones 3D), ¿cuántos grupos "naturales" consideran que existen en este dataset de clientes, y qué método usarían para definirlos?**

No necesariamente tiene que ganar el método con el Silhouette Score más alto — como ya discutieron en el ejercicio de Iris, el mejor score no siempre es sinónimo de la mejor solución para el problema de negocio. Justifiquen su elección considerando también la interpretabilidad y la forma de los grupos.

**A partir de aquí, van a trabajar con el método y el número de clusters que acaban de seleccionar.**

---

## 9. Fase 7 — Interpretación de los clusters (perfilado)

Esta es la parte más "de negocio" del workshop. Tener clusters no sirve de nada si no sabemos **qué representa cada uno**. Vamos a perfilar cada cluster usando las variables originales (sin escalar, para que los números tengan sentido real: años, dólares, puntaje).

### 9.1. Agreguen la etiqueta de cluster al DataFrame original

```python
df["Cluster"] = labels  # las labels del método que seleccionaron en la Fase 6
```

### 9.2. Extraigan métricas por cluster

```python
df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].median()
df.groupby("Cluster")["Genre"].value_counts(normalize=True)
```

### 9.3. Boxplots por variable, separados por cluster

```python
import seaborn as sns

for col in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Cluster", y=col)
    plt.title(f"{col} por cluster")
    plt.show()
```

### 9.4. Construyan el "perfil" de cada cluster

Con base en las métricas y los boxplots, describan cada cluster en una frase, como en el ejemplo que vimos en clase: *"si en un cluster el score de gasto es alto y el ingreso también, podríamos decir que representa a clientes de clase alta con alto poder adquisitivo."*

Completen una tabla como esta (ejemplo con 2 clusters, ajústenla al número real que encontraron):

| Cluster | Edad promedio | Ingreso promedio | Spending Score promedio | Interpretación / "persona" |
|---:|---:|---:|---:|---|
| 0 | ... | ... | ... | *(ej: "Clientes jóvenes de ingreso medio-alto y gasto alto — probablemente el segmento más rentable para campañas de marketing")* |
| 1 | ... | ... | ... | ... |

> **Importante:** la interpretación debe salir de los números, no al revés. No le pongan una etiqueta bonita a un cluster si las métricas no la respaldan — describan primero lo que ven, y después pónganle nombre.

### Pregunta

- ¿Alguno de los clusters les llamó la atención por combinar valores que no esperaban (ej. ingreso alto pero gasto bajo)? ¿Qué estrategia de negocio le darían ustedes a ese grupo?

---

## 10. Fase 8 — El cluster como variable objetivo

Ahora viene el giro del workshop: vamos a usar la columna `Cluster` que acaban de crear como **variable objetivo (target)** para un problema de clasificación supervisada.

¿Por qué hacemos esto? Porque en la práctica, el clustering es costoso de recalcular cada vez que llega un cliente nuevo (hay que volver a correr el algoritmo sobre todo el dataset). Si entrenamos un clasificador que **aprende a predecir el cluster** a partir de las características del cliente, podemos asignarle un segmento a un cliente nuevo de forma instantánea, sin tener que re-ejecutar el clustering completo.

```python
X = df[["Genre_encoded", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
y = df["Cluster"]
```

---

## 11. Fase 9 — Entrenen un clasificador

Elijan **un** clasificador de esta lista (el que prefieran, no hay una respuesta "correcta"):

- KNN
- Decision Tree
- Random Forest
- SVM

> **Nota:** si estaban pensando en usar Regresión Lineal, recuerden que `Cluster` es una variable **categórica** (aunque esté representada con números 0, 1, 2...), no continua — para este problema necesitan un modelo de **clasificación**. Si quieren algo en el espíritu de un modelo lineal, usen **Regresión Logística** en su lugar.

### 11.1. División train/test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # <- piensen: ¿por qué conviene usar stratify aquí?
)
```

### 11.2. Entrenamiento

> **Pista de estructura (ejemplo con KNN — adapten según el modelo que elijan):**
> ```python
> from sklearn.neighbors import KNeighborsClassifier
> from sklearn.preprocessing import StandardScaler
> from sklearn.pipeline import Pipeline
>
> pipeline = Pipeline([
>     ("scaler", StandardScaler()),
>     ("clf", KNeighborsClassifier(n_neighbors=...))  # <- completen
> ])
>
> pipeline.fit(X_train, y_train)
> y_pred = pipeline.predict(X_test)
> ```

Recuerden: cualquier modelo basado en distancias (KNN, SVM) necesita las variables escaladas; los basados en árboles (Decision Tree, Random Forest) no lo requieren, pero no les hace daño incluirlo en el pipeline de todas formas.

---

## 12. Fase 10 — Evalúen el desempeño del clasificador

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Preguntas de cierre

1. ¿Qué tan bien logró el clasificador predecir el cluster de un cliente nuevo?
2. Si el accuracy fue muy alto (cercano a 1.0), ¿qué les dice eso sobre qué tan "separables" son los clusters que encontraron en el espacio de características originales?
3. Si el clasificador se equivocó mucho entre dos clusters en particular, revisen la matriz de confusión: ¿esos dos clusters tenían perfiles parecidos en la Fase 7? ¿Tiene sentido que el modelo los confunda?

---

## 13. Reflexión final

Para cerrar, respondan en equipo (no hace falta extenderse, 3-4 líneas por pregunta):

> **¿El método de clustering que obtuvo el mayor Silhouette Score fue el que terminaron usando para definir los segmentos de negocio? ¿Por qué sí o por qué no?**

> **Si el centro comercial les pidiera, en una sola frase por segmento, una recomendación de marketing para cada cluster que encontraron, ¿qué les dirían?**

¡Buen trabajo! Este flujo completo — clustering, interpretación con datos reales, y conversión a un problema supervisado — es exactamente el tipo de proyecto que podrían encontrarse en un caso real de analítica de clientes.
