from flask import Flask, render_template
import clustering
from spark import obtener_resultados

app = Flask(__name__)

@app.route('/')
def home():
    info = clustering.RealizarClustering()
    return render_template("index.html", data=info)

@app.route("/spark/")
def resultadosSpark():
    resultados = obtener_resultados()
    
    return render_template(
        "Spark.html",
        ventas_ciudad=resultados["ventas_ciudad"],
        ventas_categoria=resultados["ventas_categoria"],
        promedio_tienda=resultados["promedio_tienda"]
    )


if __name__ == "__main__":
    app.run(debug=True)
