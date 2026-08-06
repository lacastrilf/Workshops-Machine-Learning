# Data cleaning — World Happiness Report 2015–2019

**Fecha:** 2026-08-06
**Entregable:** `word_happines.ipynb`

## Problema

Los cinco CSV del World Happiness Report (2015–2019) usan esquemas distintos: nombres
de columna diferentes para la misma variable, columnas que existen solo en algunos años,
y una columna `Region` presente únicamente en 2015 y 2016. Un `pd.concat` directo produce
782 filas × 31 columnas mayormente nulas, que es el estado actual del notebook.

## Esquema objetivo

11 columnas, 782 filas:

| Columna | Origen 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|
| `year` | derivada | derivada | derivada | derivada | derivada |
| `Overall rank` | `Happiness Rank` | `Happiness Rank` | `Happiness.Rank` | `Overall rank` | `Overall rank` |
| `Country` | `Country` | `Country` | `Country` | `Country or region` | `Country or region` |
| `Region` | `Region` | `Region` | *propagada* | *propagada* | *propagada* |
| `Score` | `Happiness Score` | `Happiness Score` | `Happiness.Score` | `Score` | `Score` |
| `GDP per Capita` | `Economy (GDP per Capita)` | idem | `Economy..GDP.per.Capita.` | `GDP per capita` | `GDP per capita` |
| `Social support` | `Family` | `Family` | `Family` | `Social support` | `Social support` |
| `Health (Life Expectancy)` | `Health (Life Expectancy)` | idem | `Health..Life.Expectancy.` | `Healthy life expectancy` | idem |
| `Freedom` | `Freedom` | `Freedom` | `Freedom` | `Freedom to make life choices` | idem |
| `Generosity` | `Generosity` | `Generosity` | `Generosity` | `Generosity` | `Generosity` |
| `Perceptions of corruption` | `Trust (Government Corruption)` | idem | `Trust..Government.Corruption.` | `Perceptions of corruption` | idem |

**Columnas eliminadas:** `Standard Error` (2015), `Lower/Upper Confidence Interval` (2016),
`Whisker.high`/`Whisker.low` (2017), `Dystopia Residual` (2015–2017).

## Decisión: `Family` ≡ `Social support`

Se fusionan en una sola columna llamada `Social support`.

Ambas son la misma variable del Gallup World Poll — el promedio nacional de respuestas
binarias a *"If you were in trouble, do you have relatives or friends you can count on to
help you whenever you need them, or not?"*. El WHR la etiquetó `Family` de 2015 a 2017 y
la renombró a `Social support` desde el reporte de 2018, porque "Family" era engañoso: la
pregunta incluye amigos. Es un renombre, no un cambio de variable.

**Nota de comparabilidad (documentar en el notebook, no corregir):** los valores de
`GDP per Capita`, `Social support`, `Health`, `Freedom`, `Generosity` y `Perceptions of
corruption` no son valores crudos, sino la *contribución explicada* (valor crudo ×
coeficiente de regresión de ese año). Los coeficientes varían por año, así que los valores
no son estrictamente comparables entre años. Por eso se conserva `year`.

## Decisión: propagación de `Region`

`Region` solo existe en 2015 y 2016 (164 países únicos). Se construye un mapa
`Country → Region` desde la unión de ambos años y se propaga a 2017–2019 con `.map()`.

Un merge ingenuo deja 8 nulos por nombres de país inconsistentes. Se normalizan antes:

| Nombre a corregir | Nombre canónico |
|---|---|
| `Hong Kong S.A.R., China` | `Hong Kong` |
| `Taiwan Province of China` | `Taiwan` |
| `Northern Cyprus` | `North Cyprus` |
| `Trinidad & Tobago` | `Trinidad and Tobago` |
| `North Macedonia` | `Macedonia` |
| `Somaliland region` | `Somaliland Region` |

`Gambia` (aparece solo en 2019) no existe en 2015/2016. Se asigna manualmente a
`Sub-Saharan Africa`.

Resultado esperado: `Region` sin nulos en las 782 filas.

## Decisión: valores faltantes

`Perceptions of corruption` de **United Arab Emirates, 2018** viene vacío en el CSV
original. Es el único nulo real del dataset limpio. **Se deja como `NaN`** y se documenta.
La imputación, si hace falta, corresponde a la etapa de modelado, no a la de limpieza.

## Pipeline

Cinco pasos, una celda por paso, cada una precedida de markdown explicativo:

1. **Renombrar columnas** — un diccionario por año → esquema estándar.
2. **Normalizar nombres de país** — aplicar el diccionario de renombres a los 5 dataframes.
3. **Propagar `Region`** — construir el mapa desde 2015+2016, aplicarlo a 2017–2019,
   añadir `Gambia` a mano.
4. **Eliminar columnas** fuera del esquema objetivo.
5. **Concatenar** con `ignore_index=True` y reordenar columnas al orden del esquema.

## Verificación

Celdas finales que comprueban:

- `df.shape == (782, 11)`
- `df["Region"].isna().sum() == 0`
- `df.groupby("year").size()` → 158, 157, 155, 156, 156
- `df.isna().sum()` → un solo nulo, en `Perceptions of corruption`
- `df.info()` y `df.head()`

## Fuera de alcance

Normalización de escalas entre años, imputación, feature engineering, EDA y visualización.
Este spec cubre únicamente la limpieza y consolidación.
