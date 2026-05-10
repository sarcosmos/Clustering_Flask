
import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, desc

# Ruta absoluta al CSV de ventas. Usar Path evita errores si Flask se ejecuta
# desde una carpeta distinta a la raiz del proyecto.
RUTA_CSV = Path(__file__).parent / "data" / "ventas.csv"

# Spark necesita saber que interprete de Python usar para el driver y los workers.
# Se usa el mismo ejecutable actual para respetar el entorno virtual del proyecto.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark_session():
    """Crea o reutiliza una sesion local de Spark para procesar el CSV."""
    # local[1] ejecuta Spark en un solo hilo, suficiente para este proyecto y mas
    # estable en una aplicacion Flask local.
    spark = SparkSession.builder \
        .appName("AnalisisVentasLocal") \
        .master("local[1]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()

    return spark


def cargar_datos():
    """Lee ventas.csv y agrega la columna calculada total_venta."""
    spark = get_spark_session()

    # header=True usa la primera fila como nombres de columnas.
    # inferSchema=True permite que Spark detecte numeros en vez de leer todo como texto.
    df = spark.read.csv(
        str(RUTA_CSV),
        header=True,
        inferSchema=True
    )

    # total_venta representa el ingreso por fila: unidades vendidas por precio unitario.
    # Esta columna se usa luego para sumar o promediar ventas por grupo.
    df = df.withColumn(
        "total_venta",
        col("cantidad") * col("precio_unitario")
    )

    return df


def obtener_resultados():
    """Calcula los resumenes que se muestran en la plantilla Spark.html."""
    df = cargar_datos()

    # Agrupa por ciudad para identificar donde se concentra el mayor volumen de ventas.
    ventas_ciudad = df.groupBy("ciudad") \
        .agg(sum("total_venta").alias("total_ventas")) \
        .orderBy(desc("total_ventas")) \
        .toPandas() \
        .to_dict(orient="records")

    # Agrupa por categoria para comparar que tipos de productos generan mas ingresos.
    ventas_categoria = df.groupBy("categoria") \
        .agg(sum("total_venta").alias("total_ventas")) \
        .orderBy(desc("total_ventas")) \
        .toPandas() \
        .to_dict(orient="records")

    # Calcula el promedio por tienda para ver el desempeno tipico de cada punto de venta.
    promedio_tienda = df.groupBy("tienda") \
        .agg(avg("total_venta").alias("promedio_venta")) \
        .orderBy(desc("promedio_venta")) \
        .toPandas() \
        .to_dict(orient="records")

    # Flask trabaja comodamente con listas de diccionarios, por eso cada resultado
    # se convierte desde Spark DataFrame a Pandas y luego a records.
    return {
        "ventas_ciudad": ventas_ciudad,
        "ventas_categoria": ventas_categoria,
        "promedio_tienda": promedio_tienda
    }
