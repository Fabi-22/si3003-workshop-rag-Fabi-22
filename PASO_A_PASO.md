# Control de ejecución del workshop RAG

Fuente: https://ia-eafit-rag.netlify.app/setup y sus páginas Guía, Tareas y Entrega.
Usuario: Fabi-22. Correo: fvalenciab@eafit.edu.co.
Estado: preparación iniciada; aún no hay tareas acreditadas ni resultados propios.

## 1. Preparación

- [ ] Iniciar sesión en GitHub como Fabi-22.
- [ ] Crear `si3003-workshop-rag-Fabi-22` con README. La guía recomienda público para revisión.
- [ ] Subir el contenido de la plantilla, incluido `.gitignore`, y confirmar el commit.
- [ ] Iniciar sesión en NVIDIA y generar la API key.
- [ ] Guardarla en Secrets de Colab como `NVIDIA_API_KEY` y habilitar acceso por notebook.
- [ ] Ejecutar instalación y configuración del 02b; verificar los mensajes reales.
- [ ] Guardar copias personales de ambos notebooks y hacer commits frecuentes.

## 2. Teoría — 10 puntos

- [ ] T1: calcular capas ≥1, ≥2 y ≥3 para M=16 y N=1.000.000; simular en Python; comparar y explicar hubs.
- [ ] T2: explicar k=10 con ef=5, valor inicial recomendado por la presentación y ef frente a ef_construction.
- [ ] T3: completar tabla inserción/búsqueda con justificación por fila y relacionar add_documents/invoke.
- [ ] T4: tabla de posiciones antes/después de borrar B sin actualizar FAISS; identificar errores silenciosos y excepción; relación con Qdrant.
- [ ] T5: calcular costo de 100 consultas/segundo, notación científica y compromiso de ANN.

## 3. Notebook 02b — 20 puntos

- [ ] A1 (4): ejecutar celdas en orden, lanzar Gradio, probar 4 ejemplos y 1 pregunta sin respuesta; tabla de 5 respuestas y fuentes; `evidencias/A1_gradio.png`.
- [ ] A2 (6): crear `data/company_documents.json` con ≥8 documentos, id/title/content/metadata y dos claves de metadata; cargar ruta correcta; probar 5 preguntas respondibles y 2 fuera de contexto; tabla con corrección; justificar chunk_size; `evidencias/A2_corpus.png`.
- [ ] A3 (5): guardar respuestas a pregunta fácil, ambigua y fuera de contexto; modificar prompt con citas [título], frase fija de abstención y ≤3 oraciones; repetir; tabla, diff y conclusión con resultados.
- [ ] A4 (5): retornar vectorstore; mostrar scores usando similarity_search_with_score(question,k=3); probar 3 buenas y 2 fuera de contexto; tabla y umbral justificado; `evidencias/A4_scores.png`.

## 4. Notebook 03 — 30 puntos

- [ ] B1 (6): ejecutar partes 0–4, verificar chunks en Qdrant; Q1–Q5 y Bonus; tabla de 6 con respuesta, corrección y paper; imprimir fuentes bonus y comprobar ambos papers.
- [ ] B2 (8): usar funciones auxiliares del taller; construir exactamente tres índices con chunk_size=400/900/1500 y overlap≈13%; reutilizar 900 para k=3/5/8 y umbral=0.15/0.25/0.40; tabla de 9 configuraciones con chunks, aciertos de 5 y observación; conclusión de 5–8 líneas.
- [ ] B3 (6): 3 preguntas plausibles sin respuesta; clasificar abstención/invención/mezcla; si inventa, corregir prompt, repetir y mostrar diff.
- [ ] B4 (5): traducir Q1/Q3/Q4; comparar chunks y calidad EN/ES; hipótesis y método de comprobación.
- [ ] B5 (5): leer PDFs; 3 preguntas nuevas con ground truth y sección, una combinando papers; bloques details, ejecución y tabla.

## 5. Reflexión — 10 puntos

- [ ] R1: cinco respuestas de ≥3 oraciones cada una, apoyadas en resultados propios: contexto insuficiente, fuentes bonus, parámetros, modelos separados y paso a producción.

## 6. Retos — máximo 30 puntos

Hay 70 puntos opcionales disponibles; el total acreditable de retos es 30. No marcar retos sin evidencia.

- [ ] C1 (15): FAISS exacto/HNSW, M={8,16,32}, efSearch={10,16,32,64,128,256}; recall@10, latencia y construcción; gráficas; n={20.000,100.000,500.000}; conclusiones.
- [ ] C2 (12): paper HNSW arXiv 1603.09320; reindexar; 3 preguntas propias con ground truth + 1 cruzada; contrastar presentaciones.
- [ ] C3 (15): Gradio de papers con títulos/archivo/score y Dropdown por paper; filtro Qdrant metadata.source; comparar con/sin filtro; captura.
- [ ] C4 (12): CSV de 10 preguntas, fuente esperada y palabra clave; hit_at_k; hit@3/hit@5 para los tres tamaños; dos limitaciones.
- [ ] C5 (8): MMR k=5/fetch_k=20/lambda_mult=0.5; comparar 3 preguntas con similitud; cobertura, redundancia y conclusión.
- [ ] C6 (8): índice Qdrant en disco; indexar una vez, reiniciar runtime y reabrir sin reindexar; verificar respuesta y comparar tiempos; límites de producción.

## 7. Entrega

- [ ] Completar nombre y código estudiantil en README.
- [ ] Mantener README con tareas realmente completadas, puntos y enlaces.
- [ ] Completar RESPUESTAS.md con tablas, fuentes y análisis revisados por la estudiante.
- [ ] Guardar notebooks ejecutados en notebooks/ y capturas reales en evidencias/.
- [ ] Revisar archivos y salidas para evitar API keys; si hubo exposición, revocar.
- [ ] Confirmar commits del 21 de octubre de 2026 y acceso del profesor al repo.
- [ ] Preparar correo institucional a jipadillab@eafit.edu.co con nombre, código, enlace, tareas, retos y autoevaluación.
- [ ] Enviar hoy al finalizar la clase. Confirmar el envío real antes de marcarlo.

Los puntos publicados son una estimación; la calificación final corresponde al profesor.
