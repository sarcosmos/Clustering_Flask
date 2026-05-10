import csv
import os
from pathlib import Path
from statistics import median

# Limita procesos paralelos de scikit-learn en equipos locales para evitar advertencias
# o consumo excesivo de CPU al ejecutar el modelo desde Flask.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RUTA_CSV = Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Columnas que se tratan como numeros reales. Se convierten manualmente porque el
# CSV llega como texto y K-Means necesita datos numericos.
COLUMNAS_NUMERICAS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Estas columnas no se usan para entrenar: customerID solo identifica al cliente,
# y Churn se reserva para interpretar el resultado despues del agrupamiento.
COLUMNAS_EXCLUIDAS = ["customerID", "Churn"]

# Paleta usada por las graficas SVG para distinguir visualmente cada cluster.
COLORES_CLUSTER = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b", "#0891b2"]

# Informacion que se muestra en la vista para explicar que variables entran al modelo
# y por que son utiles para segmentar clientes.
VARIABLES_USADAS = [
    {
        "variable": "gender",
        "significado": "Sexo registrado del cliente: Female o Male.",
        "uso": "Ayuda a detectar diferencias de comportamiento entre grupos demograficos.",
    },
    {
        "variable": "SeniorCitizen",
        "significado": "Indica si el cliente es adulto mayor: 1 si lo es, 0 si no.",
        "uso": "Permite que el modelo considere segmentos por edad avanzada.",
    },
    {
        "variable": "Partner",
        "significado": "Indica si el cliente tiene pareja.",
        "uso": "Puede relacionarse con estabilidad o permanencia en el servicio.",
    },
    {
        "variable": "Dependents",
        "significado": "Indica si el cliente tiene personas dependientes.",
        "uso": "Ayuda a identificar perfiles familiares o individuales.",
    },
    {
        "variable": "tenure",
        "significado": "Cantidad de meses que el cliente lleva con la empresa.",
        "uso": "Es clave para separar clientes nuevos, intermedios y antiguos.",
    },
    {
        "variable": "PhoneService",
        "significado": "Indica si el cliente tiene servicio telefonico.",
        "uso": "Describe el tipo de producto contratado.",
    },
    {
        "variable": "MultipleLines",
        "significado": "Indica si el cliente tiene multiples lineas telefonicas.",
        "uso": "Diferencia clientes con servicios simples o mas completos.",
    },
    {
        "variable": "InternetService",
        "significado": "Tipo de servicio de internet: DSL, Fiber optic o sin internet.",
        "uso": "Suele marcar grupos con cargos y necesidades distintas.",
    },
    {
        "variable": "OnlineSecurity",
        "significado": "Indica si tiene seguridad en linea.",
        "uso": "Representa servicios adicionales contratados.",
    },
    {
        "variable": "OnlineBackup",
        "significado": "Indica si tiene respaldo en linea.",
        "uso": "Aporta informacion sobre paquetes digitales adicionales.",
    },
    {
        "variable": "DeviceProtection",
        "significado": "Indica si tiene proteccion de dispositivo.",
        "uso": "Ayuda a separar clientes con mayor nivel de servicio.",
    },
    {
        "variable": "TechSupport",
        "significado": "Indica si tiene soporte tecnico.",
        "uso": "Puede distinguir clientes con planes mas completos.",
    },
    {
        "variable": "StreamingTV",
        "significado": "Indica si tiene servicio de television por streaming.",
        "uso": "Representa consumo de servicios de entretenimiento.",
    },
    {
        "variable": "StreamingMovies",
        "significado": "Indica si tiene servicio de peliculas por streaming.",
        "uso": "Complementa el perfil de servicios de entretenimiento.",
    },
    {
        "variable": "Contract",
        "significado": "Tipo de contrato: mensual, un anio o dos anios.",
        "uso": "Es importante porque contratos largos suelen asociarse con menor abandono.",
    },
    {
        "variable": "PaperlessBilling",
        "significado": "Indica si usa facturacion electronica.",
        "uso": "Describe preferencias de facturacion.",
    },
    {
        "variable": "PaymentMethod",
        "significado": "Metodo de pago del cliente.",
        "uso": "Ayuda a encontrar patrones asociados a pagos automaticos o manuales.",
    },
    {
        "variable": "MonthlyCharges",
        "significado": "Cargo mensual del cliente.",
        "uso": "Separa clientes por nivel de gasto recurrente.",
    },
    {
        "variable": "TotalCharges",
        "significado": "Cargo total acumulado durante la relacion con la empresa.",
        "uso": "Resume valor historico y antiguedad economica del cliente.",
    },
]


def ObtenerDatos():
    """Lee el dataset y prepara los valores numericos basicos."""
    with RUTA_CSV.open(newline="", encoding="utf-8") as archivo:
        datos = list(csv.DictReader(archivo))

    # TotalCharges tiene algunos valores vacios. Se calcula la mediana para rellenarlos
    # sin distorsionar tanto el conjunto como podria hacerlo un cero.
    total_charges_validos = [
        float(fila["TotalCharges"])
        for fila in datos
        if fila["TotalCharges"].strip()
    ]
    total_charges_mediana = median(total_charges_validos)

    # K-Means no puede trabajar con texto en columnas numericas, por eso se convierten
    # a float antes de crear la matriz del modelo.
    for fila in datos:
        for columna in COLUMNAS_NUMERICAS:
            valor = fila[columna].strip()
            fila[columna] = float(valor) if valor else total_charges_mediana

    return datos


def PrepararDatos(datos):
    """Convierte filas del CSV en una matriz numerica lista para K-Means."""
    variables = []

    for fila in datos:
        fila_modelo = {}
        for columna, valor in fila.items():
            if columna in COLUMNAS_EXCLUIDAS:
                continue

            if columna in COLUMNAS_NUMERICAS:
                fila_modelo[columna] = valor
            else:
                # Las categorias se guardan como "columna=valor" para que DictVectorizer
                # cree columnas one-hot y el modelo pueda comparar opciones como contrato o pago.
                fila_modelo[columna] = f"{columna}={valor}"

        variables.append(fila_modelo)

    # DictVectorizer convierte los diccionarios de cada cliente en una tabla numerica.
    vectorizer = DictVectorizer(sparse=False)
    variables_vectorizadas = vectorizer.fit_transform(variables)

    # StandardScaler pone todas las columnas en una escala comparable. Esto evita que
    # cargos monetarios grandes dominen variables pequenas como SeniorCitizen.
    scaler = StandardScaler()
    variables_escaladas = scaler.fit_transform(variables_vectorizadas)

    return variables_escaladas, vectorizer.get_feature_names_out().tolist()


def RealizarClustering(n_clusters=5):
    """Ejecuta el flujo completo y devuelve datos listos para el template HTML."""
    datos = ObtenerDatos()
    variables_escaladas, columnas_modelo = PrepararDatos(datos)

    # random_state hace que los resultados sean repetibles en cada ejecucion.
    # n_init=10 prueba varios puntos iniciales para buscar clusters mas estables.
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(variables_escaladas)
    silhouette = CrearSilhouetteScore(variables_escaladas, etiquetas)

    # Se agrega la etiqueta de cluster a cada fila original para poder mostrarla
    # junto con los datos interpretables del cliente.
    for fila, etiqueta in zip(datos, etiquetas):
        fila["cluster"] = int(etiqueta)

    resumen_cluster = CrearResumen(datos)

    return {
        "resultado": datos[:25],
        "resumen_cluster": resumen_cluster,
        "centroides": CrearCentroides(datos),
        "grafica_clusters": CrearGraficaClusters(datos, variables_escaladas, modelo),
        "grafica_codo": CrearGraficaCodo(variables_escaladas),
        "graficas_negocio": CrearGraficasNegocio(resumen_cluster),
        "interpretacion": CrearInterpretacion(datos),
        "metricas": CrearMetricasRapidas(datos, resumen_cluster, n_clusters),
        "silhouette": silhouette,
        "columnas_modelo": columnas_modelo,
        "variables_usadas": VARIABLES_USADAS,
        "total_clientes": len(datos),
        "clientes_mostrados": min(25, len(datos)),
        "n_clusters": n_clusters,
    }


def CrearResumen(datos):
    """Calcula promedios y conteos por cluster para resumir el resultado."""
    resumen = {}

    for fila in datos:
        cluster = fila["cluster"]
        if cluster not in resumen:
            resumen[cluster] = {
                "cluster": cluster,
                "clientes": 0,
                "antiguedad_total": 0,
                "cargo_mensual_total": 0,
                "cargo_total_total": 0,
                "churn_si": 0,
                "churn_no": 0,
            }

        grupo = resumen[cluster]
        grupo["clientes"] += 1
        grupo["antiguedad_total"] += fila["tenure"]
        grupo["cargo_mensual_total"] += fila["MonthlyCharges"]
        grupo["cargo_total_total"] += fila["TotalCharges"]
        if fila["Churn"] == "Yes":
            grupo["churn_si"] += 1
        else:
            grupo["churn_no"] += 1

    resultado = []
    for grupo in resumen.values():
        clientes = grupo["clientes"]
        # Se dividen los acumulados entre la cantidad de clientes para obtener
        # perfiles promedio faciles de comparar entre clusters.
        resultado.append({
            "cluster": grupo["cluster"],
            "clientes": clientes,
            "antiguedad_promedio": round(grupo["antiguedad_total"] / clientes, 2),
            "cargo_mensual_promedio": round(grupo["cargo_mensual_total"] / clientes, 2),
            "cargo_total_promedio": round(grupo["cargo_total_total"] / clientes, 2),
                "churn_si": grupo["churn_si"],
                "churn_no": grupo["churn_no"],
                "churn_porcentaje": round(grupo["churn_si"] * 100 / clientes, 2),
            })

    return sorted(resultado, key=lambda grupo: grupo["cluster"])


def CrearSilhouetteScore(variables_escaladas, etiquetas):
    """Calcula y clasifica el Silhouette Score del agrupamiento."""
    valor = float(silhouette_score(variables_escaladas, etiquetas))

    if valor < 0.25:
        nivel = "Malo"
        clase = "malo"
        mensaje = "Los grupos tienen separacion debil y pueden mezclarse entre si."
    elif valor < 0.50:
        nivel = "Aceptable"
        clase = "aceptable"
        mensaje = "Los grupos muestran una separacion util, aunque todavia hay solapamientos."
    else:
        nivel = "Bueno"
        clase = "bueno"
        mensaje = "Los grupos estan bien separados y son mas faciles de interpretar."

    return {
        "valor": round(valor, 3),
        "nivel": nivel,
        "clase": clase,
        "mensaje": mensaje,
    }


def CrearMetricasRapidas(datos, resumen, n_clusters):
    """Construye indicadores ejecutivos para la parte superior del dashboard."""
    total_clientes = len(datos)
    churn_promedio = sum(1 for fila in datos if fila["Churn"] == "Yes") * 100 / total_clientes
    cargo_mensual_promedio = sum(fila["MonthlyCharges"] for fila in datos) / total_clientes
    mayor_churn = max(resumen, key=lambda grupo: grupo["churn_porcentaje"])
    mas_estable = min(resumen, key=lambda grupo: grupo["churn_porcentaje"])

    return [
        {"titulo": "Total clientes", "valor": f"{total_clientes:,}", "detalle": "Registros analizados"},
        {"titulo": "Clusters", "valor": n_clusters, "detalle": "Segmentos K-Means"},
        {"titulo": "Churn promedio", "valor": f"{churn_promedio:.1f}%", "detalle": "Base completa"},
        {"titulo": "Mayor churn", "valor": f"C{mayor_churn['cluster']}", "detalle": f"{mayor_churn['churn_porcentaje']:.1f}% de abandono"},
        {"titulo": "Mas estable", "valor": f"C{mas_estable['cluster']}", "detalle": f"{mas_estable['churn_porcentaje']:.1f}% de abandono"},
        {"titulo": "Cargo mensual", "valor": f"${cargo_mensual_promedio:,.0f}", "detalle": "Promedio general"},
    ]


def CrearGraficasNegocio(resumen):
    """Prepara barras SVG para las visualizaciones por cluster."""
    return {
        "churn": CrearBarras(resumen, "churn_porcentaje", "%", "Churn por cluster"),
        "tamano": CrearBarras(resumen, "clientes", "", "Clientes por cluster"),
        "antiguedad": CrearBarras(resumen, "antiguedad_promedio", " meses", "Antiguedad promedio"),
        "cargos": CrearBarras(resumen, "cargo_mensual_promedio", "", "Cargo mensual promedio", moneda=True),
    }


def CrearBarras(resumen, campo, sufijo, titulo, moneda=False):
    """Escala valores de resumen a una grafica de barras horizontal en SVG."""
    maximo = max(grupo[campo] for grupo in resumen) or 1
    barras = []

    for indice, grupo in enumerate(resumen):
        valor = grupo[campo]
        if moneda:
            etiqueta = f"${valor:,.0f}"
        elif isinstance(valor, float):
            etiqueta = f"{valor:.1f}{sufijo}"
        else:
            etiqueta = f"{valor}{sufijo}"

        barras.append({
            "cluster": grupo["cluster"],
            "valor": valor,
            "etiqueta": etiqueta,
            "ancho": round((valor / maximo) * 100, 2),
            "color": COLORES_CLUSTER[indice % len(COLORES_CLUSTER)],
        })

    return {"titulo": titulo, "barras": barras}


def CrearCentroides(datos):
    """Resume cada cluster como un centroide interpretable en la escala original."""
    resumen = CrearResumen(datos)
    return [
        {
            "cluster": grupo["cluster"],
            "tenure": grupo["antiguedad_promedio"],
            "MonthlyCharges": grupo["cargo_mensual_promedio"],
            "TotalCharges": grupo["cargo_total_promedio"],
            "churn_porcentaje": round(grupo["churn_si"] * 100 / grupo["clientes"], 2),
        }
        for grupo in resumen
    ]


def CrearGraficaClusters(datos, variables_escaladas, modelo):
    """Prepara puntos SVG para visualizar clusters en dos dimensiones."""
    # PCA reduce todas las columnas del modelo a dos componentes. No reemplaza al
    # entrenamiento; solo permite dibujar una aproximacion visual en la pagina.
    pca = PCA(n_components=2, random_state=42)
    puntos_pca = pca.fit_transform(variables_escaladas)
    centroides_pca = pca.transform(modelo.cluster_centers_)

    # Se usa una muestra para que el SVG no sea pesado cuando el dataset es grande.
    muestra = CrearMuestraPuntos(datos, puntos_pca, maximo=700)
    puntos = [
        {
            "x": float(puntos_pca[indice][0]),
            "y": float(puntos_pca[indice][1]),
            "cluster": fila["cluster"],
            "color": COLORES_CLUSTER[fila["cluster"] % len(COLORES_CLUSTER)],
        }
        for indice, fila in muestra
    ]
    centroides = [
        {
            "x": float(centroide[0]),
            "y": float(centroide[1]),
            "cluster": indice,
            "color": COLORES_CLUSTER[indice % len(COLORES_CLUSTER)],
        }
        for indice, centroide in enumerate(centroides_pca)
    ]

    return {
        "puntos": EscalarCoordenadas(puntos, centroides),
        "centroides": EscalarCoordenadas(centroides, puntos),
        "varianza": [round(valor * 100, 2) for valor in pca.explained_variance_ratio_],
    }


def CrearMuestraPuntos(datos, puntos_pca, maximo):
    """Selecciona una muestra uniforme para mantener ligera la grafica."""
    if len(datos) <= maximo:
        return list(enumerate(datos))

    paso = max(1, len(datos) // maximo)
    muestra = list(enumerate(datos))[::paso]
    return muestra[:maximo]


def EscalarCoordenadas(elementos, elementos_extra):
    """Convierte coordenadas PCA a coordenadas dentro del viewBox del SVG."""
    todos = elementos + elementos_extra
    min_x = min(elemento["x"] for elemento in todos)
    max_x = max(elemento["x"] for elemento in todos)
    min_y = min(elemento["y"] for elemento in todos)
    max_y = max(elemento["y"] for elemento in todos)
    ancho = max(max_x - min_x, 1)
    alto = max(max_y - min_y, 1)

    escalados = []
    for elemento in elementos:
        escalado = elemento.copy()
        # Los margenes 40/360 dejan espacio para ejes y etiquetas dentro del SVG.
        escalado["svg_x"] = round(40 + ((elemento["x"] - min_x) / ancho) * 720, 2)
        escalado["svg_y"] = round(360 - ((elemento["y"] - min_y) / alto) * 320, 2)
        escalados.append(escalado)

    return escalados


def CrearGraficaCodo(variables_escaladas):
    """Calcula la inercia para varios K y construye los puntos del metodo del codo."""
    inercias = []
    for k in range(1, 11):
        # La inercia mide que tan compactos son los clusters; ayuda a comparar
        # si aumentar K mejora mucho o ya aporta poco.
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(variables_escaladas)
        inercias.append({"k": k, "inercia": round(modelo.inertia_, 2)})

    min_inercia = min(punto["inercia"] for punto in inercias)
    max_inercia = max(punto["inercia"] for punto in inercias)
    rango = max(max_inercia - min_inercia, 1)

    puntos_svg = []
    for punto in inercias:
        x = 60 + ((punto["k"] - 1) / 9) * 680
        y = 360 - ((punto["inercia"] - min_inercia) / rango) * 300
        puntos_svg.append({
            "k": punto["k"],
            "inercia": punto["inercia"],
            "svg_x": round(x, 2),
            "svg_y": round(y, 2),
        })

    return {
        "puntos": puntos_svg,
        "polyline": " ".join(f"{p['svg_x']},{p['svg_y']}" for p in puntos_svg),
    }


def CrearInterpretacion(datos):
    """Genera lectura de negocio comparando cada cluster contra el promedio general."""
    resumen = CrearResumen(datos)
    promedio_tenure = sum(fila["tenure"] for fila in datos) / len(datos)
    promedio_mensual = sum(fila["MonthlyCharges"] for fila in datos) / len(datos)
    churn_total = sum(1 for fila in datos if fila["Churn"] == "Yes") * 100 / len(datos)

    interpretaciones = []
    for grupo in resumen:
        churn_grupo = grupo["churn_si"] * 100 / grupo["clientes"]
        antiguedad = DescribirNivel(grupo["antiguedad_promedio"], promedio_tenure)
        cargo = DescribirNivel(grupo["cargo_mensual_promedio"], promedio_mensual)
        churn = DescribirNivel(churn_grupo, churn_total)
        perfil = ClasificarPerfil(antiguedad, cargo, churn)
        interpretaciones.append({
            "cluster": grupo["cluster"],
            "perfil": perfil["perfil"],
            "comportamiento": (
                f"Agrupa {grupo['clientes']} clientes con antiguedad {antiguedad}, "
                f"cargo mensual {cargo} y churn {churn} frente al promedio general."
            ),
            "riesgo": perfil["riesgo"],
            "estabilidad": perfil["estabilidad"],
            "gasto": perfil["gasto"],
            "estrategia": perfil["estrategia"],
        })

    return interpretaciones


def ClasificarPerfil(antiguedad, cargo, churn):
    """Traduce patrones numericos a perfiles de negocio accionables."""
    if churn == "alto" and cargo == "alto":
        return {
            "perfil": "Clientes premium en riesgo",
            "riesgo": "Alto: combinan buen ingreso mensual con probabilidad elevada de abandono.",
            "estabilidad": "Requieren seguimiento cercano porque su permanencia no esta consolidada.",
            "gasto": "Alto valor recurrente para la empresa.",
            "estrategia": "Priorizar retencion, beneficios personalizados y revision de experiencia de servicio.",
        }

    if churn == "alto":
        return {
            "perfil": "Clientes con alto riesgo",
            "riesgo": "Alto: el abandono supera el comportamiento promedio de la base.",
            "estabilidad": "Baja o media, segun su antiguedad dentro del servicio.",
            "gasto": f"Nivel de gasto {cargo}; conviene medir rentabilidad antes de incentivar.",
            "estrategia": "Aplicar campanas de recuperacion, soporte proactivo y ofertas de permanencia.",
        }

    if antiguedad == "alto" and churn == "bajo":
        return {
            "perfil": "Clientes fidelizados",
            "riesgo": "Bajo: muestran permanencia y menor tendencia de abandono.",
            "estabilidad": "Alta; son una base adecuada para programas de lealtad.",
            "gasto": f"Nivel de gasto {cargo}, con oportunidad de venta cruzada si el valor es medio o bajo.",
            "estrategia": "Mantener satisfaccion, ofrecer upgrades graduales y reconocer la antiguedad.",
        }

    if cargo == "alto":
        return {
            "perfil": "Clientes rentables",
            "riesgo": f"Riesgo {churn}; deben monitorearse por su impacto en ingresos.",
            "estabilidad": f"Antiguedad {antiguedad}, lo que define el tipo de acompanamiento.",
            "gasto": "Alto; aportan ingresos mensuales superiores al promedio.",
            "estrategia": "Cuidar experiencia, soporte y bundles para proteger el valor del segmento.",
        }

    if antiguedad == "bajo":
        return {
            "perfil": "Clientes nuevos",
            "riesgo": f"Riesgo {churn}; estan en una etapa temprana de relacion.",
            "estabilidad": "Todavia en formacion, sensibles a problemas iniciales de servicio.",
            "gasto": f"Nivel de gasto {cargo}.",
            "estrategia": "Mejorar onboarding, educacion del producto y contactos tempranos de satisfaccion.",
        }

    return {
        "perfil": "Clientes de bajo valor o comportamiento medio",
        "riesgo": f"Riesgo {churn}; no son el segmento mas critico pero conviene observar tendencias.",
        "estabilidad": f"Antiguedad {antiguedad}, con margen para fortalecer la relacion.",
        "gasto": f"Nivel de gasto {cargo}.",
        "estrategia": "Usar ofertas segmentadas de bajo costo y evaluar oportunidades de upgrade.",
    }


def DescribirNivel(valor, promedio):
    """Clasifica un valor como alto, medio o bajo frente a un promedio."""
    if valor > promedio * 1.15:
        return "alto"
    if valor < promedio * 0.85:
        return "bajo"
    return "medio"
