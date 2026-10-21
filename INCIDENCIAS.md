# Incidencias reales de ejecución

- El modelo generativo del material, `nvidia/nemotron-3-super-120b-a12b`, devolvió HTTP 404 para la cuenta institucional. La búsqueda del catálogo NVIDIA mostró `nvidia/nemotron-3.5-super-vl-120b-a12b`; se usó ese modelo para continuar y se verificó una respuesta correcta sobre el precio Enterprise. Se conserva el modelo de embeddings original. Los resultados no deben compararse como si se hubieran obtenido con la versión original.
- Se añadió `.strip()` a la lectura del Secret para eliminar espacios/saltos de línea accidentales. La clave permanece en Secrets; no se incluye en los archivos.
- Los documentos demo produjeron 4 chunks en esta ejecución, frente a 5 en las salidas originales del curso. Las salidas originales se eliminaron de las copias locales antes de ejecutarlas.

- Por indicación explícita de la estudiante, la configuración final usa `nvidia/llama-nemotron-embed-vl-1b-v2` como embeddings. Se repiten los experimentos dependientes del índice con ese modelo; los resultados preliminares quedan separados.
- NVIDIA devolvió un error 500 temporal en la comparación del prompt. Se incorporan reintentos limitados sin cambiar la pregunta ni inventar resultados.

- La nueva versión generativa puede agotar el presupuesto de 700/800 tokens sin producir contenido visible. La muestra oficial de NVIDIA documenta `chat_template_kwargs.enable_thinking`; se prueba `False` para mantener respuestas visibles y concisas con ese presupuesto. Se mantiene el mismo ajuste dentro de cada comparación. Fuente: https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b (ejemplo LangChain).
