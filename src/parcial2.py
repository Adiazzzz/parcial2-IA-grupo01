from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
ARCHIVO = DATA / "grupo_01.csv"
ARCHIVO_LIMPIO = DATA / "grupo_01_limpio.csv"
UMBRAL_KM = 100  


def linea(titulo):
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)

# TAREA 2. Carga e inspección de datos.

def cargar_datos(ruta):
   
    try:
        df = pd.read_csv(ruta)
        print(f"Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df
    except FileNotFoundError:
        print(f"Error: el archivo '{ruta}' no existe.")
        return None


def explorar_datos(df):
    
    print("\nPRIMERAS 5 FILAS:")
    print(df.head())

    print(f"\nFORMA DEL DATAFRAME): {df.shape}")

    print("\nINFORMACIÓN GENERAL:")
    df.info()

    print("\nTIPOS DE DATOS (dtypes):")
    print(df.dtypes)

    print("\nNULOS POR COLUMNA:")
    print(df.isnull().sum())

    print(f"\nFILAS DUPLICADAS: {df.duplicated().sum()}")

    
    
    print("\nPREGUNTA CLAVE - TIPOS DE LAS COLUMNAS NUMÉRICAS:")
    for col in ["altitud_msnm", "lluvia_mm_mes"]:
        print(f"  {col}: dtype = {df[col].dtype}")

   
    con_decimales = df[df["altitud_msnm"] % 1 != 0]
    print(f"\nAltitudes con decimales (debieran ser metros enteros): {len(con_decimales)}")
    print(con_decimales[["finca_id", "altitud_msnm"]])
    print(f"Rango de altitud sin esos valores: "
          f"{df.loc[df['altitud_msnm'] % 1 == 0, 'altitud_msnm'].min():.0f} a "
          f"{df.loc[df['altitud_msnm'] % 1 == 0, 'altitud_msnm'].max():.0f} m")
    print("Conclusión: altitud_msnm es float64 porque hay valores en kilómetros "
          "mezclados con metros; lluvia_mm_mes es float64 por sus decimales y nulos (normal).")

#TAREA 3. Limpieza y calidad de datos. 

def detectar_problemas(df):
   
    linea("TAREA 3 - DETECCIÓN DE PROBLEMAS DE CALIDAD")

    #  duplicados
    duplicadas = df[df.duplicated(keep=False)].sort_values("finca_id")
    print(f"\n[1] FILAS DUPLICADAS: {df.duplicated().sum()} filas repetidas "
          f"(ids repetidos: {df['finca_id'].duplicated().sum()})")
    print(duplicadas)

    # valores faltantes
    print(f"\n[2] VALORES FALTANTES: {df.isnull().sum().sum()} en total")
    print(df[df.isnull().any(axis=1)])

    #  valor con unidades equivocadas (kilómetros en vez de metros)
    sospechosos = df[df["altitud_msnm"] < UMBRAL_KM]
    print(f"\n[3] ALTITUD CON UNIDADES EQUIVOCADAS: {len(sospechosos)} valor(es) < {UMBRAL_KM}")
    print(sospechosos)
    print(f"    Resto de altitudes: {df.loc[df['altitud_msnm'] >= UMBRAL_KM, 'altitud_msnm'].min():.0f}"
          f" a {df['altitud_msnm'].max():.0f} m")

    # categorías escritas de formas distintas
    print(f"\n[4] CATEGORÍAS INCONSISTENTES EN 'municipio': "
          f"{df['municipio'].nunique()} variantes escritas")
    print({repr(k): int(v) for k, v in df["municipio"].value_counts().items()})
    normalizado = df["municipio"].str.strip().str.title()
    print(f"    Tras quitar espacios y unificar mayúsculas: {normalizado.nunique()} municipios")

    # Revisiones sin hallazgos (para dejar constancia de que se verificaron)
    print("\nOTRAS REVISIONES (sin hallazgos):")
    print(f"  Filas de resumen (TOTAL, PROMEDIO...): "
          f"{df['finca_id'].str.upper().str.contains('TOTAL|PROM|SUMA|MEDIA').sum()}")
    print(f"  Texto dentro de columnas numéricas: "
          f"{sum(not pd.api.types.is_numeric_dtype(df[c]) for c in ['altitud_msnm', 'lluvia_mm_mes'])}")
    print(f"  Lluvia <= 0: {(df['lluvia_mm_mes'] <= 0).sum()} | Altitud <= 0: "
          f"{(df['altitud_msnm'] <= 0).sum()}")
    print(f"  Valores de la columna objetivo: {sorted(df['cosecha_buena'].unique().tolist())}")


def limpiar_datos(df):
    """Corrige cada problema y retorna el DataFrame limpio."""
    linea("TAREA 3 - LIMPIEZA")
    limpio = df.copy()
    print(f"Filas iniciales: {len(limpio)}")

    # 1.unificar el nombre de los municipios
    limpio["municipio"] = limpio["municipio"].str.strip().str.title()
    print(f"[municipio] unificado: {limpio['municipio'].nunique()} municipios -> "
          f"{sorted(limpio['municipio'].unique())}")

    # 2.  convertir de kilómetros a metros
    mascara = limpio["altitud_msnm"] < UMBRAL_KM
    for idx in limpio[mascara].index:
        antes = limpio.loc[idx, "altitud_msnm"]
        limpio.loc[idx, "altitud_msnm"] = round(antes * 1000)
        print(f"[altitud] {limpio.loc[idx, 'finca_id']}: {antes} km -> "
              f"{limpio.loc[idx, 'altitud_msnm']:.0f} m")

    # 3. conservar la primera aparición (datos duplicados)
    n = len(limpio)
    limpio = limpio.drop_duplicates()
    print(f"[duplicados] eliminadas {n - len(limpio)} filas -> {len(limpio)} filas")

    # 4.  eliminar las fincas sin lluvia (variable de entrada)
    n = len(limpio)
    ids_sin_lluvia = limpio.loc[limpio["lluvia_mm_mes"].isnull(), "finca_id"].tolist()
    limpio = limpio.dropna(subset=["lluvia_mm_mes"])
    print(f"[nulos] eliminadas {n - len(limpio)} filas sin lluvia {ids_sin_lluvia} "
          f"-> {len(limpio)} filas")

    # La altitud ya son metros enteros: se deja como entero
    limpio["altitud_msnm"] = limpio["altitud_msnm"].astype(int)
    return limpio.reset_index(drop=True)


def validar_limpieza(limpio):
   
    assert limpio.isnull().sum().sum() == 0, "quedan nulos"
    assert limpio.duplicated().sum() == 0, "quedan duplicados"
    assert limpio["finca_id"].is_unique, "quedan ids repetidos"
    assert limpio["municipio"].nunique() == 5, "municipios sin unificar"
    assert limpio["altitud_msnm"].min() >= 1000, "altitud en unidades equivocadas"
    print("\nValidación final: sin nulos, sin duplicados, 5 municipios, altitudes en metros.")


def resumen_limpieza(original, limpio):
    
    linea("RESUMEN DE LA LIMPIEZA")
    print(f"Filas antes: {len(original)} | filas después: {len(limpio)} "
          f"(se eliminaron {len(original) - len(limpio)})")
    antes = original["cosecha_buena"].value_counts().sort_index()
    despues = limpio["cosecha_buena"].value_counts().sort_index()
    tabla = pd.DataFrame({"antes": antes, "despues": despues})
    tabla.index.name = "cosecha_buena"
    print("\nRegistros por clase (0 = deficiente, 1 = buena):")
    print(tabla)
    print(f"\nProporción de cosechas buenas: antes {antes[1] / len(original):.1%} | "
          f"después {despues[1] / len(limpio):.1%}")


def pregunta_clave_media(original, limpio):
    
    linea("PREGUNTA CLAVE - ALTITUD (EL VALOR MÁS SOSPECHOSO: F032 = 1.654)")
    col = "altitud_msnm"

    # Mismas 84 filas, cambiando solo F032: aísla el efecto de la corrección
    corregida = original[col].where(original[col] >= UMBRAL_KM, original[col] * 1000)

    media_antes, std_antes = original[col].mean(), original[col].std(ddof=0)
    media_corr, std_corr = corregida.mean(), corregida.std(ddof=0)
    print(f"Media sin corregir (84 filas originales):   {media_antes:8.2f} m | "
          f"desviación {std_antes:7.2f}")
    print(f"Media corrigiendo solo F032 (mismas filas): {media_corr:8.2f} m | "
          f"desviación {std_corr:7.2f}")
    print(f"Media del dataset limpio ({len(limpio)} filas):      "
          f"{limpio[col].mean():8.2f} m | desviación {limpio[col].std(ddof=0):7.2f}")

    # Efecto sobre la estandarización z = (x - media) / desviación
    z_malo = (original.loc[original["finca_id"] == "F032", col].iloc[0] - media_antes) / std_antes
    z_resto = (original.loc[original[col] >= UMBRAL_KM, col] - media_antes) / std_antes
    z_ok = (limpio[col] - limpio[col].mean()) / limpio[col].std(ddof=0)
    print(f"\nEstandarización SIN corregir: F032 queda en z = {z_malo:.2f} y las demás "
          f"fincas solo ocupan z de {z_resto.min():.2f} a {z_resto.max():.2f}")
    print(f"Estandarización corregida:    todas las fincas quedan entre z = "
          f"{z_ok.min():.2f} y {z_ok.max():.2f}")


def main():
    print("=" * 60)
    print("PARCIAL 2 - FINCAS CAFETERAS DE CARTAGO (GRUPO 01)")
    print("=" * 60)

    df = cargar_datos(ARCHIVO)
    if df is None:
        return

    explorar_datos(df)

    detectar_problemas(df)
    limpio = limpiar_datos(df)
    validar_limpieza(limpio)
    resumen_limpieza(df, limpio)
    pregunta_clave_media(df, limpio)

    limpio.to_csv(ARCHIVO_LIMPIO, index=False)
    print(f"\nDataset limpio guardado en '{ARCHIVO_LIMPIO}'")


if __name__ == "__main__":
    main()