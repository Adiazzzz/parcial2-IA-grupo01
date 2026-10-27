# Parcial Segundo Corte - Inteligencia Artificial (Grupo 01)

**Integrantes:** Manuela Aguirre Toro y Alejandra Díaz Velásquez
**Archivo asignado:** `grupo_01.csv` (Café: fincas cafeteras de la región de Cartago)

## Tarea 1. Configuración del entorno

Entorno con Python 3.12, NumPy, Pandas, Matplotlib y Seaborn, dentro de un contenedor Docker llamado `parcial2_ia` (para no chocar con `python_ia` de las clases). No se usa Scikit-Learn ni ninguna otra librería de Machine Learning: el perceptrón está hecho solo con NumPy.

Comandos para reproducir el entorno (en WSL, desde la carpeta del proyecto):

```bash
mkdir -p parcial2/src parcial2/data
cd parcial2
docker ps                          # si python_ia está encendido: docker stop python_ia
docker-compose up -d --build
docker exec -it parcial2_ia bash
```

Dentro del contenedor:

```bash
cd src
python parcial2.py
```

Los gráficos quedan en la carpeta `graficos/` y el CSV limpio en `data/grupo_01_limpio.csv`.

---

## Tarea 2. Carga e inspección de datos

El script `src/parcial2.py` carga el CSV con Pandas y muestra en consola las primeras 5 filas, la forma (`shape`), los tipos de datos (`info()` y `dtypes`), los nulos por columna y las filas duplicadas.

Resultados del archivo original:

- Forma: **84 filas y 5 columnas** (`finca_id`, `municipio`, `altitud_msnm`, `lluvia_mm_mes`, `cosecha_buena`).
- Nulos: **4 valores faltantes**, todos en `lluvia_mm_mes`.
- Filas duplicadas: **4**.

**Pregunta clave: ¿alguna columna numérica aparece con un tipo de dato que no debería tener? ¿Por qué ocurre?**

`altitud_msnm` aparece como `float64`, cuando una altitud en metros debería ser un entero (`int64`). Ocurre porque hay un valor, el de la finca F032, que está escrito en kilómetros (`1.654`) mezclado con valores en metros (`1220` a `1994`). Ese único decimal obliga a Pandas a guardar toda la columna como número decimal. `lluvia_mm_mes` también es `float64`, pero eso es normal: tiene decimales y 4 nulos.

---

## Tarea 3. Calidad y limpieza de datos con Pandas

Se encontraron **4 problemas de calidad**.

### Problema 1: filas duplicadas
- **Cómo se detectó:** `df.duplicated().sum()` devolvió 4 filas repetidas, y se mostraron con `df[df.duplicated(keep=False)]`.
- **Decisión:** eliminar los duplicados y conservar la primera aparición (`drop_duplicates()`).
- **Por qué:** una finca repetida cuenta dos veces y le da más peso del que le corresponde en el entrenamiento.

### Problema 2: valores faltantes en `lluvia_mm_mes`
- **Cómo se detectó:** `df.isnull().sum()` mostró 4 nulos en `lluvia_mm_mes`. Una de las filas con nulo estaba también entre las duplicadas, así que tras quitar duplicados quedaron 3 filas sin lluvia (F010, F026 y F054).
- **Decisión:** eliminar esas 3 filas (`dropna`).
- **Por qué:** la lluvia es una de las dos entradas del perceptrón y no se puede entrenar con valores vacíos. Inventar un valor (por ejemplo, la media) metería datos que no se midieron, y con solo 3 filas la pérdida de información es pequeña.

### Problema 3: unidades equivocadas en `altitud_msnm`
- **Cómo se detectó:** `df["altitud_msnm"] < 100` devolvió un único registro: F032 con altitud `1.654`, mientras el resto va de 1220 a 1994 m.
- **Decisión:** convertir de kilómetros a metros (`1.654 × 1000 = 1654 m`).
- **Por qué:** la finca es válida y su valor es plausible en metros. Eliminarla perdería un dato bueno, y dejarla así la haría parecer una finca a nivel del mar.

### Problema 4: categorías de `municipio` escritas de formas distintas
- **Cómo se detectó:** `df["municipio"].value_counts()` mostró **11 variantes** para solo 5 municipios, por ejemplo `EL ÁGUILA`, `el águila`, `'El Águila '` (con espacio), `alcalá`, `' Alcalá'` y `CARTAGO`.
- **Decisión:** quitar espacios y unificar mayúsculas con `.str.strip().str.title()`.
- **Por qué:** sin unificar, un mismo municipio se contaría como varios y los gráficos y conteos saldrían mal. Quedaron los 5 municipios: Alcalá, Ansermanuevo, Cartago, El Águila y Ulloa.


### Filas y clases antes y después

| | Antes | Después |
|---|---|---|
| Filas | 84 | 77 |
| Clase 0 (cosecha deficiente) | 48 | 45 |
| Clase 1 (cosecha buena) | 36 | 32 |

La proporción de cosechas buenas pasó de 42.9 % a 41.6 %, así que la limpieza no desbalanceó las clases. 


### Pregunta clave: ¿qué valor le pareció más sospechoso?

El valor más sospechoso fue la altitud de **F032 = 1.654**, que es un valor imposible para una finca cafetera si se lee en metros.

| Altitud | Media | Desviación |
|---|---|---|
| Sin corregir (84 filas originales) | 1620.28 m | 297.57 |
| Corrigiendo solo F032 (mismas 84 filas) | 1639.95 m | 238.72 |
| Dataset limpio (77 filas) | 1633.22 m | 235.33 |

**Cómo habría afectado el escalado:** el perceptrón estandariza con `z = (x - media) / desviación`. Sin corregir, F032 quedaría en z = -5.44, y las demás fincas solo ocuparían z entre -1.35 y 1.26, porque la desviación inflada (297.57 contra 238.72) las comprime. El modelo vería casi todas las altitudes muy parecidas entre sí y el peso de la altitud saldría mal calculado. Con la corrección, todas las fincas quedan entre z = -1.76 y 1.53.

---

## Tarea 4. Visualización con Seaborn

### Gráfico 1: dispersión de altitud contra lluvia (`graficos/grafico1_dispersion.png`)
Muestra cada finca como un punto, con la altitud en el eje X, la lluvia en el eje Y, y el color según la cosecha (verde = buena, rojo = deficiente).

Las fincas con cosecha buena se agrupan arriba y a la derecha (más altitud y más lluvia), y las deficientes abajo y a la izquierda. Las dos clases se pueden separar bastante bien con una línea. La altitud media es 1825 m en las fincas buenas contra 1497 m en las deficientes, y la lluvia media es 220.6 mm contra 162.1 mm.

### Gráfico 2: fincas por municipio según la cosecha (`graficos/grafico2_countplot_municipio.png`)
Gráfico de barras con el número de fincas con cosecha buena y deficiente en cada municipio.

| Municipio | Deficiente (0) | Buena (1) |
|---|---|---|
| Alcalá | 13 | 4 |
| Ansermanuevo | 7 | 8 |
| Cartago | 12 | 5 |
| El Águila | 8 | 13 |
| Ulloa | 5 | 2 |

La cosecha buena es más frecuente en El Águila (13 contra 8) y está casi pareja en Ansermanuevo (8 contra 7). En Alcalá (13 contra 4) y en Cartago (12 contra 5) predominan las cosechas deficientes. Aun así, todos los municipios tienen fincas de ambas clases, así que el municipio por sí solo no explica la cosecha. Ulloa tiene solo 7 fincas, por lo que no permite sacar conclusiones firmes.

---

## Tarea 5. Perceptrón desde cero con NumPy

Implementado en `src/parcial2.py` solo con NumPy:

- Se separan `X` (altitud y lluvia) e `y` (cosecha buena) como arreglos de NumPy.
- Se estandariza `X` (media 0 y desviación 1 por columna) y se guardan la media y la desviación: media = [1633.22, 186.41], desviación = [235.33, 61.96].
- Funciones: `sigmoide`, `predecir_proba` (`X @ w + b`), `perdida_logaritmica` y `exactitud`.
- Entrenamiento con descenso del gradiente durante **500 épocas** y tasa de aprendizaje 0.1, registrando la pérdida y la exactitud en cada época.

Resultados:

| | Valor |
|---|---|
| Pesos iniciales | w1 = 0.0305, w2 = -0.1040, b = 0.0000 |
| Pesos finales | w1 = 2.9355, w2 = 2.0171, b = -0.9211 |
| Pérdida inicial | 0.7081 |
| Pérdida final | 0.1818 |
| Exactitud final | 93.51 % |

Gráficos generados:
- `graficos/curva_aprendizaje.png`: la pérdida baja de 0.7081 a 0.1818 a lo largo de las épocas, lo que indica que el modelo aprendió.
- `graficos/frontera_decision.png`: puntos de las dos clases y la línea de la frontera, dibujada en las unidades originales (metros y mm/mes).

---

## Tarea 6. Interpretación de resultados

### 1. Signo de w1 y w2
Los dos pesos son positivos (w1 = 2.9355 para la altitud y w2 = 2.0171 para la lluvia): a mayor altitud y a mayor lluvia, sube la probabilidad de una cosecha buena. El peso de la altitud es mayor, así que pesa más en la decisión. Tiene sentido para Cartago y su zona cafetera: el café de ladera alta madura más lento y de mejor calidad, y la lluvia aporta el agua que necesita el cultivo. Los datos lo confirman: la altitud media es 1825 m en las fincas con cosecha buena contra 1497 m en las deficientes, y la lluvia media es 220.6 mm contra 162.1 mm.

### 2. Separabilidad
Sí, los datos son aproximadamente linealmente separables. En `frontera_decision.png` las fincas buenas quedan casi todas arriba y a la derecha de la línea, y las deficientes abajo y a la izquierda. Del lado equivocado quedan **5 de 77 registros** (exactitud de 93.51 %): 2 fincas deficientes que el modelo clasifica como buenas (F009, con 1979 m pero solo 105 mm de lluvia, y F071) y 3 fincas buenas que clasifica como deficientes (F016, F027 y F031). F031 es un caso llamativo: tiene solo 1396 m, pero 278.4 mm de lluvia y cosecha buena, así que la lluvia alta compensó la baja altitud, algo que el modelo lineal solo captura en parte.

### 3. Dos registros nuevos
Se escalaron con la media y la desviación del entrenamiento (altitud: 1633.22 y 235.33; lluvia: 186.41 y 61.96).

| Registro | Altitud | Lluvia | z altitud | z lluvia | Probabilidad de cosecha buena | Esperado |
|---|---|---|---|---|---|---|
| Finca A | 1850 m | 200 mm | 0.92 | 0.22 | 0.903 | 1 |
| Finca B | 1300 m | 120 mm | -1.42 | -1.07 | 0.001 | 0 |

El resultado coincide con lo esperado: la finca A, alta y con buena lluvia, tiene 90.3 % de probabilidad de cosecha buena, y la finca B, baja y seca, tiene 0.1 %.

### 4. Recomendación para la entidad local
Se recomienda priorizar el apoyo técnico (asistencia, riego y manejo del cultivo) en las fincas de menos de 1500 m de altitud y menos de 160 mm de lluvia al mes. En nuestros datos hay 9 fincas con esas dos condiciones y las 9 tuvieron cosecha deficiente. Además, las fincas deficientes tienen una altitud media de 1497 m y una lluvia media de 162.1 mm, frente a 1825 m y 220.6 mm en las buenas. El modelo acierta el 93.51 % de los casos, así que puede usarse para marcar qué fincas están en riesgo antes de la cosecha.

### 5. Limitaciones del modelo
- Solo hay 77 fincas después de limpiar, y algunos municipios tienen muy pocas (por ejemplo, Ulloa con 7).
- Usa solo dos variables. Faltan suelo, variedad del café, plagas y manejo del cultivo.
- La exactitud de 93.51 % se midió con los mismos datos del entrenamiento, sin una parte reservada para validar. Con fincas nuevas puede ser menor.
- La frontera es una línea recta, y la relación real puede no serlo, como muestran los 5 registros mal clasificados.
- Muestra correlación, no causa: subir la altitud no es algo que una finca pueda hacer.
- Un error tiene costo real: una finca deficiente clasificada como buena se quedaría sin apoyo.
