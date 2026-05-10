from pathlib import Path

import pandas as pd


RUTA_CSV = Path(__file__).parent / "data" / "ventas.csv"


def cargar_datos():
    """Lee ventas.csv y agrega la columna calculada total_venta."""
    df = pd.read_csv(RUTA_CSV, encoding="latin-1")
    df["total_venta"] = df["cantidad"] * df["precio_unitario"]
    return df


def obtener_resultados():
    """Calcula los resumenes que se muestran en la plantilla Spark.html."""
    df = cargar_datos()

    ventas_ciudad = (
        df.groupby("ciudad", as_index=False)["total_venta"]
        .sum()
        .rename(columns={"total_venta": "total_ventas"})
        .sort_values("total_ventas", ascending=False)
        .to_dict(orient="records")
    )

    ventas_categoria = (
        df.groupby("categoria", as_index=False)["total_venta"]
        .sum()
        .rename(columns={"total_venta": "total_ventas"})
        .sort_values("total_ventas", ascending=False)
        .to_dict(orient="records")
    )

    promedio_tienda = (
        df.groupby("tienda", as_index=False)["total_venta"]
        .mean()
        .rename(columns={"total_venta": "promedio_venta"})
        .sort_values("promedio_venta", ascending=False)
        .to_dict(orient="records")
    )

    return {
        "ventas_ciudad": ventas_ciudad,
        "ventas_categoria": ventas_categoria,
        "promedio_tienda": promedio_tienda,
    }
