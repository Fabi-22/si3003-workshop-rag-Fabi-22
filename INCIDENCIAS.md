# Incidencias reales de ejecución

- El modelo generativo del material, `nvidia/nemotron-3-super-120b-a12b`, devolvió HTTP 404 para la cuenta institucional. La búsqueda del catálogo NVIDIA mostró `nvidia/nemotron-3.5-super-vl-120b-a12b`; se usó ese modelo para continuar y se verificó una respuesta correcta sobre el precio Enterprise. En esa prueba preliminar se conservó el embedding original; la configuración final se describe abajo. Los resultados no deben compararse como si se hubieran obtenido con la versión original.
- Se añadió `.strip()` a la lectura del Secret para eliminar espacios/saltos de línea accidentales. La clave permanece en Secrets; no se incluye en los archivos.
- Los documentos demo produjeron 4 chunks en esta ejecución, frente a 5 en las salidas originales del curso. Las salidas originales se eliminaron de las copias locales antes de ejecutarlas.

- Por indicación explícita de la estudiante, la configuración final usa `nvidia/llama-nemotron-embed-vl-1b-v2` como embeddings. Se repitieron los experimentos dependientes del índice con ese modelo. Los archivos de evidencia entregados contienen únicamente los resultados finales.
- NVIDIA devolvió un error 500 temporal en la comparación del prompt. Se incorporan reintentos limitados sin cambiar la pregunta ni inventar resultados.

- La nueva versión generativa puede agotar el presupuesto de 700/800 tokens sin producir contenido visible. La muestra oficial de NVIDIA documenta `chat_template_kwargs.enable_thinking`; se usó `False` para mantener respuestas visibles y concisas con ese presupuesto. Se mantiene el mismo ajuste dentro de cada comparación. Fuente: https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b (ejemplo LangChain).

- La instalación de 03 mostró avisos de dependencias de requests/pillow y del tipo del nuevo embedding. La extracción de ambos PDFs, los 150 chunks, los vectores de 2048 dimensiones, las consultas y la interfaz se ejecutaron correctamente; los avisos originales se conservan en las salidas.
- La copia directa de 03 desde Colab a la ruta del repositorio falló. Se recuperó la exportación nativa con sus conteos y salidas y se subió mediante la interfaz web de GitHub; se verificó después que los archivos públicos coincidieran byte a byte con los entregables locales.
