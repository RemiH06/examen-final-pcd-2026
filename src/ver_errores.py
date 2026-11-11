"""Muestra en la terminal los FP y FN del champion en prueba para parafrasearlos a mano.

Imprime mensajes originales del corpus: es sólo para leerlos localmente. No copies su salida
al informe, a MLflow, a Git ni a una IA (sección 5); en el informe van paráfrasis propias.

Uso: `uv run python src/ver_errores.py`
"""

import informe

familia = informe.seleccion()[0]["familia"]
pred = informe.predicciones_test()
tabla = informe.errores(pred, familia)
for fila in tabla.itertuples():
    print(f"\n[{fila.id_prueba}] {fila.tipo}\n  {pred.loc[fila.id_prueba, 'text']}")
