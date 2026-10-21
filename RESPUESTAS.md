# Respuestas del workshop RAG
Fabiola Valencia Barrios · 1000352691 · Fabi-22

Resultados de ejecuciones del 21 de octubre de 2026. Las tablas se contrastan con las salidas de Colab y los registros JSON enlazados en cada bloque.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ [simulación reproducible](notebooks/01_teoria_hnsw.ipynb) y [salida real](evidencias/T1_simulacion.txt).

Para N=1.000.000 y M=16, E[N≥k]=N·16^(-k). A mano: capa ≥1 = 1.000.000/16 = 62.500; capa ≥2 = 1.000.000/256 = 3.906,25; capa ≥3 = 1.000.000/4096 = 244,140625. Los valores fraccionarios son esperanzas, no cantidades observadas.

| Capa mínima | Esperado | Simulado (semilla 22) | Diferencia |
|---|---:|---:|---:|
| 1 | 62500 | 62954 | 454 |
| 2 | 3906,25 | 3894 | -12,25 |
| 3 | 244,140625 | 260 | 15,859375 |

La simulación se ejecutó localmente con un millón de muestras y la fórmula del enunciado; las diferencias son compatibles con variación aleatoria. Pocos hubs en las capas altas permiten orientar la búsqueda con pocas comparaciones y saltos grandes. Al bajar, se explora una región más cercana a la consulta en vez de recorrer todos los vectores.

Nota de lectura: los porcentajes ilustrativos 37%/14% de la presentación no corresponden a mL=1/ln(16); para este ejercicio se usa explícitamente M^(-k), como pide la tarea.

## T2 · ef, k y el tamaño de la cola

Fuente: [presentación de búsqueda](https://ia-eafit-rag.netlify.app/teoria/hnsw_busqueda).

1. Con k=10 y ef=5, el conjunto final tiene capacidad para cinco candidatos; no permite elegir diez vecinos. Por eso se exige ef≥k; algunas librerías pueden ajustar el valor internamente, pero la configuración conceptual es inconsistente.
2. La presentación propone empezar con ef=2·k (20 para k=10) y aumentarlo si el recall medido es insuficiente. Más exploración puede recuperar vecinos mejores a costa de latencia.
3. Se puede cambiar ef durante las consultas sin reconstruir el índice. ef_construction interviene al elegir conexiones durante la inserción; cambiarlo después no rehace las aristas que ya se construyeron.

## T3 · Inserción frente a búsqueda

Fuente: presentaciones de [inserción](https://ia-eafit-rag.netlify.app/teoria/hnsw_insercion) y [búsqueda](https://ia-eafit-rag.netlify.app/teoria/hnsw_busqueda).

| Aspecto | Inserción | Búsqueda | Por qué |
|---|---|---|---|
| ¿Se sortea la capa? | Sí, se asigna altura al nuevo nodo. | No, se usa la jerarquía existente. | La inserción decide en qué niveles existirá el vector; la consulta no agrega un nodo. |
| ef en capas altas | 1 en el descenso por encima del nivel asignado. | 1 durante el descenso hasta la capa 1. | Basta un candidato para encontrar rápidamente una zona de entrada. |
| Parámetro principal | ef_construction en niveles donde se conecta el nodo. | ef en la capa 0. | Una operación elige conexiones para construir el grafo; la otra mantiene candidatos para responder. |
| ¿Modifica el índice? | Sí: agrega vectores y relaciones. | No: devuelve vecinos. | Consultar no debe cambiar la estructura por cada pregunta. |

En el notebook 03, vectorstore.add_documents(...) inserta documentos y vectores, mientras retriever.invoke(...) consulta los ya almacenados. Esta tabla describe HNSW; Qdrant local en memoria puede resolver búsquedas mediante su implementación local exacta y no constituye por sí solo una medición de HNSW.

## T4 · El invariante de FAISSVectorStore

Fuente: [FAISSVectorStore explicado](https://ia-eafit-rag.netlify.app/teoria/faiss_explicado).

| Posición FAISS | Documento antes | self.documents después de eliminar B | Efecto si se recupera |
|---:|---|---|---|
| 0 | A | A | Sigue alineado. |
| 1 | B | C | Error silencioso: el vector de B se interpreta como C. |
| 2 | C | D | Error silencioso: el vector de C se interpreta como D. |
| 3 | D | No existe | IndexError al acceder a documents[3]. |

Para los índices [1,2,3], las dos primeras lecturas producirían C y D incorrectamente; al procesar 3 se lanza una excepción, así que una función sin manejo de errores no llega a devolver una lista completa. El problema es la pérdida de sincronización entre posiciones vectoriales y documentos. Una base vectorial como Qdrant administra identificadores, payloads y borrados de puntos conjuntamente, evitando tener que mantener a mano dos listas paralelas.

## T5 · El costo de la fuerza bruta

Fuente: [Small World y HNSW](https://ia-eafit-rag.netlify.app/teoria/hnsw_explainer).

768.000.000 multiplicaciones/consulta × 100 consultas/segundo = 76.800.000.000 multiplicaciones/segundo = **7,68×10^10 multiplicaciones/s**. A esto se suman sumas, lectura de memoria y selección de vecinos. Buscar de manera exacta sigue siendo posible con recursos suficientes, pero el costo crece linealmente con el corpus y la concurrencia, dificultando mantener baja latencia con un presupuesto limitado. ANN sacrifica la garantía de recuperar siempre los vecinos exactos a cambio de reducir trabajo y tiempo de consulta.

---

# Bloque A · App RAG empresarial

Configuración final: generación `nvidia/nemotron-3.5-super-vl-120b-a12b`, embeddings `nvidia/llama-nemotron-embed-vl-1b-v2`. Se configuró `chat_template_kwargs={"enable_thinking": False}` después de observar respuestas vacías con el presupuesto original. Los errores y cambios están documentados en [INCIDENCIAS.md](INCIDENCIAS.md).

## A1 · Ejecuta la app RAG de punta a punta

Evidencias: [notebook ejecutado](notebooks/02b_rag_app_nvidia_nim_colab.ipynb), [captura de Gradio](evidencias/A1_gradio.png), [resultados completos](evidencias/resultados_A.json). La ejecución final produjo **4 documentos y 4 chunks**. Se ejecutaron instalación, configuración, carga, pipeline e interfaz; la celda `gr.close_all()` se reserva para detener instancias, como indica el material.

| Pregunta | Respuesta real | Fuentes recuperadas | Evaluación |
|---|---|---|---|
| ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | El plan Enterprise cuesta $49/mes por usuario con compromiso anual. | CloudSync Pro — Precios; Política de reembolsos; Política de trabajo remoto | Sí |
| ¿Cuántos días pueden trabajar remotamente los empleados? | Los empleados pueden trabajar de forma remota hasta 3 días por semana, con aprobación del jefe directo. | Política de trabajo remoto; Onboarding; Política de reembolsos | Sí |
| ¿Qué pasa con los reembolsos mayores a $100? | Requieren aprobación del supervisor y, una vez aprobados, se procesan en 3 a 5 días hábiles al método de pago original. | Política de reembolsos; CloudSync Pro — Precios; Política de trabajo remoto | Sí |
| ¿Qué ocurre durante la primera semana de onboarding? | Durante la primera semana se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding; Política de trabajo remoto; Política de reembolsos | Sí |
| ¿Cuál es el NIT legal de CloudSync Pro? | No tengo esa información en el contexto proporcionado. | CloudSync Pro — Precios; Política de trabajo remoto; Política de reembolsos | Sí: abstención correcta |

Las cuatro respuestas coinciden con los hechos del corpus, incluida la aprobación del jefe para trabajo remoto. Para el NIT, el retriever devolvió documentos relacionados con la empresa, pero ninguno contenía el dato; el modelo reconoció esa limitación. Recuperar un documento de la empresa no demuestra que ese documento responda cualquier pregunta sobre ella.

## A2 · Tu propio corpus de documentos

Evidencias: [corpus JSON](data/company_documents.json), [captura](evidencias/A2_corpus.png), notebook y resultados completos anteriores. Se creó Café Nube como empresa **ficticia**; sus precios y políticas son datos del ejercicio, no información real sobre EAFIT. Los ocho documentos contienen `id`, `title`, `content` y las claves de metadata `department` y `type`.

| Pregunta | Respuesta real | Fuentes recuperadas | ¿Correcta? |
|---|---|---|---|
| ¿Cuánto cuesta el plan Explorador y qué incluye? | El plan Explorador cuesta 82.000 pesos colombianos mensuales e incluye dos bolsas de 340 gramos de regiones distintas. Los precios incluyen IVA. | Planes de Café Nube; Taller de métodos de preparación; Programa de fidelidad Semillas | Sí |
| ¿Cuánto vale el envío de un pedido de 90000 pesos en productos a Bogotá? | El envío es gratuito, ya que el pedido supera los 80000 pesos en productos. | Cobertura y tiempos de envío; Programa de fidelidad Semillas; Planes de Café Nube | Sí |
| ¿Cuándo debo pedir una pausa para evitar el próximo cobro? | Debes solicitar la pausa con al menos 72 horas de anticipación a la fecha de renovación. | Pausa y cancelación de suscripciones; Horario y canales de soporte; Programa de fidelidad Semillas | Sí |
| ¿Cuántas personas admite el taller y cuánto cuesta por persona? | El taller admite ocho participantes y cuesta 60000 pesos por persona. | Taller de métodos de preparación; Planes de Café Nube; Cobertura y tiempos de envío | Sí |
| ¿Cuántas Semillas necesito para un descuento y de cuánto es? | Necesitas diez Semillas para obtener un descuento de 15000 pesos en productos. | Programa de fidelidad Semillas; Pausa y cancelación de suscripciones; Taller de métodos de preparación | Sí |
| ¿Cuál es el NIT legal de Café Nube? | No tengo esa información. | Planes de Café Nube; Taller de métodos de preparación; Conservación y molienda del café | Sí: no inventa |
| ¿Qué tasa de interés cobra Café Nube por financiar una compra? | No tengo esa información. | Planes de Café Nube; Pausa y cancelación de suscripciones; Taller de métodos de preparación | Sí: no inventa |
| ¿Cuánto cuesta el servicio? | No tengo esa información, ya que el contexto no especifica a qué servicio se refiere (taller, suscripción o envíos). | Taller de métodos de preparación; Planes de Café Nube; Cobertura y tiempos de envío | Abstención prudente ante la ambigüedad; no pide aclaración |

Las primeras siete filas corresponden al barrido A2; la octava incorpora la consulta ambigua registrada en `A3_before` del mismo corpus, índice y prompt inicial. Así, las tres preguntas comparadas en A3 están incluidas en esta tabla; no se presenta esta fila como una nueva ejecución.

Se mantuvo `chunk_size=400` y `chunk_overlap=40`: los documentos son breves y cada política queda completa en un solo fragmento; el resultado real fue **8 chunks**. Reducirlos artificialmente podría separar condiciones como las 72 horas de anticipación de la política de pausas. El archivo se escribe y se carga desde `Path("data/company_documents.json")`, evitando la discrepancia de rutas señalada en la guía.

## A3 · Ingeniería del prompt (antes y después)

Se mantuvieron el mismo índice, los mismos modelos, `k=3`, umbral `0.3` y configuración de inferencia; se cambió únicamente el prompt para la comparación.

```diff
--- prompt_antes
+++ prompt_despues
@@ -1 +1 @@
-Eres un asistente empresarial. Responde únicamente con información del contexto proporcionado. Si el contexto no contiene la respuesta, indica que no tienes esa información. Responde en español y de forma concisa.
+Eres un asistente empresarial. Usa únicamente el contexto proporcionado. Cita el título exacto de cada documento que respalda tu respuesta entre corchetes. Si el contexto no alcanza para responder, escribe exactamente: No tengo información suficiente en los documentos para responder. No completes datos con conocimiento externo. Responde en español con un máximo de 3 oraciones.
```

| Pregunta | Antes | Después |
|---|---|---|
| ¿Cuánto cuesta el plan Explorador y qué incluye? | El plan Explorador cuesta 82.000 pesos colombianos mensuales e incluye dos bolsas de 340 gramos de regiones distintas. Los precios incluyen IVA. | El plan Explorador cuesta 82000 pesos colombianos mensuales e incluye dos bolsas de 340 gramos de regiones distintas. [Planes de Café Nube] |
| ¿Cuánto cuesta el servicio? | No tengo esa información, ya que el contexto no especifica a qué servicio se refiere (taller, suscripción o envíos). | El taller de métodos de preparación cuesta 60000 pesos por persona. El plan Origen cuesta 45000 pesos mensuales y el plan Explorador cuesta 82000 pesos mensuales. El envío cuesta 8000 pesos por pedido y es gratuito desde 80000 pesos en productos. [Taller de métodos de preparación] [Planes de Café Nube] [Cobertura y tiempos de envío] |
| ¿Cuál es el NIT legal de Café Nube? | No tengo esa información. | No tengo información suficiente en los documentos para responder. |

La frase fija de abstención produjo el cambio más claro: la pregunta sobre el NIT pasó de una explicación libre a la frase exigida. Las respuestas respaldadas por documentos incorporaron títulos entre corchetes y la respuesta ambigua reunió precios en tres oraciones. En estas tres pruebas no se observó incumplimiento de las tres reglas; esto no garantiza que el modelo las respete siempre. La pregunta ambigua sigue siendo una limitación: el sistema enumera alternativas, pero sería más útil pedir aclaración antes de asumir qué servicio quiere el usuario. En la corrida de diagnóstico previa hubo una respuesta vacía; se conserva la incidencia y no se cuenta como respuesta válida de esta comparación final.

## A4 · Muestra el score de similitud en las fuentes

Evidencia: [interfaz con scores](evidencias/A4_scores.png). `build_pipeline` devuelve `(pipeline, chunks, vectorstore)` y `chat()` usa `vectorstore.similarity_search_with_score(question, k=3)`.

| Pregunta | ¿Hay respuesta en el corpus? | Tres fuentes y scores coseno |
|---|---|---|
| ¿Cuánto cuesta el plan Explorador y qué incluye? | Sí | Planes de Café Nube: 0.420816; Taller de métodos de preparación: 0.111132; Programa de fidelidad Semillas: 0.088339 |
| ¿Cuándo debo pedir una pausa para evitar el próximo cobro? | Sí | Pausa y cancelación de suscripciones: 0.392532; Horario y canales de soporte: 0.163624; Programa de fidelidad Semillas: 0.151738 |
| ¿Cuántas personas admite el taller y cuánto cuesta por persona? | Sí | Taller de métodos de preparación: 0.355312; Planes de Café Nube: 0.085346; Cobertura y tiempos de envío: 0.068088 |
| ¿Cuál es el NIT legal de Café Nube? | No | Planes de Café Nube: 0.396959; Taller de métodos de preparación: 0.367130; Conservación y molienda del café: 0.356720 |
| ¿Qué tasa de interés cobra Café Nube por financiar una compra? | No | Planes de Café Nube: 0.437153; Pausa y cancelación de suscripciones: 0.423581; Taller de métodos de preparación: 0.418909 |

Propongo probar **score_threshold=0.32** frente al 0.30 inicial. Los mejores scores de las tres preguntas respondibles fueron aproximadamente 0.421, 0.393 y 0.355, de modo que 0.32 retendría sus fuentes útiles y descartaría los secundarios irrelevantes de esas tres consultas. Sin embargo, las preguntas sin respuesta alcanzaron 0.397 y 0.437: ningún umbral único separa estos cinco casos, y subirlo por encima de 0.437 también descartaría las fuentes correctas de las tres preguntas respondibles. Por eso el umbral propuesto es una hipótesis para validar con más preguntas, no una defensa suficiente contra alucinaciones; se mantiene la abstención del prompt. La tabla se midió con el umbral de referencia 0.30 y la interfaz identifica los scores como similitud, no como probabilidad de que la respuesta exista.

---

# Bloque B · RAG sobre papers

Evidencias generales: [notebook ejecutado](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) y [respuestas y contextos completos](evidencias/resultados_B.json). Se usaron los dos modelos definitivos indicados en el README. Los PDFs descargados fueron validados y convertidos con MarkItDown: Attention produjo 40.174 caracteres y RAG 69.121. El índice de referencia tiene **55 + 95 = 150 chunks**, dimensión **2048** y distancia coseno.

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

| ID | Pregunta | Respuesta real | ¿Correcta? | Fragmentos recuperados | ¿Paper correcto? |
|---|---|---|---|---|---|
| Q1 | What BLEU score did the Transformer (big model) achieve on WMT 2014 English-to-German? | The Transformer (big model) achieved a BLEU score of 28.4 on WMT 2014 English-to-German. (Attention Is All You Need) | Sí | attention_paper.pdf#28; attention_paper.pdf#1; attention_paper.pdf#26; attention_paper.pdf#31; attention_paper.pdf#24 | Sí |
| Q2 | How does RAG combine parametric and non-parametric memory? | RAG combines parametric and non-parametric memory by using a pre-trained seq2seq transformer (BART) as parametric memory and a dense vector index of Wikipedia accessed by a pre-trained neural retriever (DPR) as non-parametric memory. These components are combined in a probabilistic model trained end-to-end: the retriever provides latent documents conditioned on the input, and the seq2seq model conditions on these latent documents together with the input to generate the output. Latent documents are marginalized with a top-K approximation, either per-output (same document for all tokens) or per-token (different documents for different tokens). (Retrieval-Augmented Generation \| rag_paper.pdf) | Sí | rag_paper.pdf#8; rag_paper.pdf#1; rag_paper.pdf#7; rag_paper.pdf#32; rag_paper.pdf#50 | Sí |
| Q3 | What is the purpose of the scaling factor sqrt(dk) in Scaled Dot-Product Attention? | The scaling factor √dk counteracts large dot products for large dk, which otherwise push the softmax into regions with extremely small gradients. (Attention Is All You Need) | Sí | attention_paper.pdf#11; attention_paper.pdf#12; attention_paper.pdf#13; attention_paper.pdf#16; attention_paper.pdf#10 | Sí |
| Q4 | What are the two RAG model variants proposed in the paper and how do they differ? | The two variants are RAG-Sequence and RAG-Token. RAG-Sequence uses the same retrieved document to generate each target token, treating the document as a single latent variable marginalized over top-K documents. RAG-Token can predict each target token based on a different document, allowing the generator to choose content from several documents when producing an answer. (Retrieval-Augmented Generation \| rag_paper.pdf) | Sí | rag_paper.pdf#10; rag_paper.pdf#11; rag_paper.pdf#24; rag_paper.pdf#12; rag_paper.pdf#91 | Sí |
| Q5 | How many attention heads does the base Transformer model use, and what are the key and value dimensions per head? | The base Transformer model uses 8 attention heads, with key dimension d_k = 64 and value dimension d_v = 64 per head. (Attention Is All You Need) | Sí | attention_paper.pdf#14; attention_paper.pdf#31; attention_paper.pdf#9; attention_paper.pdf#39; attention_paper.pdf#15 | Sí |
| Bonus | The RAG paper uses FAISS for fast retrieval with an HNSW approximation. The Attention paper proposes the Transformer architecture. What seq2seq model does RAG use as its generator component, and is it based on the Transformer? | The RAG paper uses BART as its generator component, and yes, it is based on the Transformer architecture. The paper states that the generator is "a pre-trained BART model" and describes BART as "a denoising autoencoder for pretraining sequence-to-sequence models" based on the Transformer architecture. (Retrieval-Augmented Generation \| rag_paper.pdf) | Sí; identifica BART y Transformer, sin precisar large | rag_paper.pdf#10; rag_paper.pdf#12; rag_paper.pdf#13; rag_paper.pdf#86; rag_paper.pdf#11 | Solo RAG; no aparecieron ambos papers |

En la bonus se recuperaron únicamente fragmentos de `rag_paper.pdf`, con índices 10, 12, 13, 86 y 11; la celda original los imprime por título y archivo. La respuesta factual es compatible con BART, un generador basado en Transformer, pero la recuperación no reunió evidencia de ambos artículos. Esto distingue corrección de la respuesta y cobertura de fuentes: no basta con que el modelo diga algo correcto para afirmar que contrastó los dos papers.

## B2 · Experimentos con chunk_size, k y score_threshold

Se copiaron `build_store` y `build_chain` de la página de tareas. Se construyeron **exactamente tres índices**: el de 900/120 de la Parte 3 se reutilizó y solo se añadieron 400/52 y 1500/195. Los cambios de k y umbral reutilizaron el índice de 900; no se recalcularon embeddings de documentos para esos seis ensayos. Cada una de las nueve configuraciones ejecutó Q1–Q5, y sus 45 respuestas y contextos están en el JSON y en la salida del notebook.

| Parámetro variado | chunk_size | overlap | k | Umbral | Chunks índice | Aciertos | Caracteres recuperados (5 preguntas) | Observación |
|---|---|---|---|---|---|---|---|---|
| chunk_size | 400 | 52 | 5 | 0.25 | 326 | 5/5 | 9024 | Cambia el número de vectores y la extensión del contexto; conserva los cinco datos correctos. |
| chunk_size | 900 | 120 | 5 | 0.25 | 150 | 5/5 | 20906 | Cambia el número de vectores y la extensión del contexto; conserva los cinco datos correctos. |
| chunk_size | 1500 | 195 | 5 | 0.25 | 96 | 5/5 | 33248 | Cambia el número de vectores y la extensión del contexto; conserva los cinco datos correctos. |
| k | 900 | 120 | 3 | 0.25 | 150 | 5/5 | 12604 | Cambia cuántos fragmentos entran al contexto; conserva los cinco datos correctos. |
| k | 900 | 120 | 5 | 0.25 | 150 | 5/5 | 20906 | Cambia cuántos fragmentos entran al contexto; conserva los cinco datos correctos. |
| k | 900 | 120 | 8 | 0.25 | 150 | 5/5 | 32908 | Cambia cuántos fragmentos entran al contexto; conserva los cinco datos correctos. |
| score_threshold | 900 | 120 | 5 | 0.15 | 150 | 5/5 | 20906 | Los cinco fragmentos por pregunta sobreviven; no cambia la recuperación en estas consultas. |
| score_threshold | 900 | 120 | 5 | 0.25 | 150 | 5/5 | 20906 | Los cinco fragmentos por pregunta sobreviven; no cambia la recuperación en estas consultas. |
| score_threshold | 900 | 120 | 5 | 0.4 | 150 | 5/5 | 20906 | Los cinco fragmentos por pregunta sobreviven; no cambia la recuperación en estas consultas. |

**Conclusión (7 líneas):**

1. Ningún parámetro cambió el total de aciertos: las nueve configuraciones lograron 5/5 en estas cinco preguntas.
2. El tamaño de fragmento sí cambió la estructura: 400 produjo 326 chunks, 900 produjo 150 y 1500 produjo 96.
3. Con k=5, los contextos de los cinco ensayos sumaron 9.024, 20.906 y 33.248 caracteres respectivamente; son caracteres, no tokens facturados.
4. Con tamaño 900, k=3 usó 12.604 caracteres frente a 32.908 con k=8, sin ganar exactitud factual en esta muestra.
5. Los umbrales 0.15, 0.25 y 0.40 conservaron los mismos cinco fragmentos por pregunta; mover el umbral no tuvo efecto observable aquí.
6. Probaría 900/120, k=3 y umbral 0.25 como candidato más económico en contexto, pero no afirmo menor costo o latencia medidos porque no se cronometraron ni facturaron esos recursos.
7. Antes de preferirlo en producción ampliaría el quiz con preguntas difíciles y cruzadas: cinco aciertos no prueban que k=3 cubra todas las necesidades.

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

| Pregunta sin respuesta en los papers | Respuesta real | Clasificación |
|---|---|---|
| What BLEU score did GPT-5 achieve when the authors compared it with the Transformer in this paper? | The paper does not mention GPT-5 or any comparison with it. The paper is "Attention Is All You Need," which focuses on the Transformer model and compares it with models such as ByteNet, Deep-Att+PosUnk, GNMT+RL, ConvS2S, and MoE on machine translation tasks. | Admite que no tiene esa información |
| How much electricity in kWh did the authors measure for training the base Transformer? | The paper does not report electricity consumption in kWh for training the base Transformer. It only reports training time (12 hours on 8 P100 GPUs) and estimated floating-point operations (FLOPS), not measured energy usage. (Attention Is All You Need) | Admite que no tiene esa información |
| What exact benchmark accuracy did the RAG authors report for Llama 4 in 2025? | The excerpts do not contain any information about Llama 4 or benchmark results from 2025. The paper discussed is "Retrieval-Augmented Generation" (rag_paper.pdf), which does not mention Llama 4. | Admite que no tiene esa información |

Las tres respuestas se abstuvieron de inventar el dato pedido. Por eso no se cambió el prompt de sistema para B3: la instrucción de la tarea era modificarlo si alguna inventaba. Las explicaciones adicionales no constituyen evidencia de una evaluación exhaustiva; la conclusión se limita a estas tres preguntas.

## B4 · Preguntas en español sobre papers en inglés

| Pregunta | Versión ES | Chunks EN | Chunks ES | Coincidencias | Respuesta ES | Calidad |
|---|---|---|---|---|---|---|
| Q1 | ¿Qué puntuación BLEU obtuvo el Transformer grande en WMT 2014 de inglés a alemán? | attention_paper.pdf#28; attention_paper.pdf#1; attention_paper.pdf#26; attention_paper.pdf#31; attention_paper.pdf#24 | attention_paper.pdf#28; attention_paper.pdf#1; attention_paper.pdf#26; attention_paper.pdf#31; attention_paper.pdf#40 | 4/5 | El Transformer grande obtuvo una puntuación BLEU de 28.4 en WMT 2014 de inglés a alemán. (Attention Is All You Need) | Correcta en EN y ES |
| Q3 | ¿Para qué sirve el factor de escala raíz cuadrada de d_k en la atención por producto escalar escalado? | attention_paper.pdf#11; attention_paper.pdf#12; attention_paper.pdf#13; attention_paper.pdf#16; attention_paper.pdf#10 | attention_paper.pdf#13; attention_paper.pdf#12; attention_paper.pdf#11; attention_paper.pdf#14; attention_paper.pdf#16 | 4/5 | El factor de escala 1/√d_k sirve para evitar que los productos punto de las consultas y claves crezcan demasiado en magnitud cuando d_k es grande. Si los valores son grandes, la función softmax entra en regiones donde tiene gradientes extremadamente pequeños, lo que dificulta el entrenamiento. Dividir por √d_k contrarresta este efecto.<br><br>Paper: Attention Is All You Need | Correcta en EN y ES |
| Q4 | ¿Cuáles son las dos variantes de RAG propuestas en el artículo y en qué se diferencian? | rag_paper.pdf#10; rag_paper.pdf#11; rag_paper.pdf#24; rag_paper.pdf#12; rag_paper.pdf#91 | rag_paper.pdf#10; rag_paper.pdf#11; rag_paper.pdf#24; rag_paper.pdf#52; rag_paper.pdf#41 | 3/5 | Las dos variantes son **RAG-Sequence** y **RAG-Token**. Se diferencian en el uso del documento recuperado: RAG-Sequence usa el mismo documento recuperado para predecir cada token de la secuencia objetivo, mientras que RAG-Token puede predecir cada token basándose en un documento diferente. (Fuente: *Retrieval-Augmented Generation*, rag_paper.pdf, Sección 2.1 Models). | Correcta en EN y ES |

Los conjuntos coincidieron en 4/5, 4/5 y 3/5 fragmentos; en Q3 también cambió el orden. La hipótesis es que `llama-nemotron-embed-vl-1b-v2` aproxima consultas semánticamente equivalentes entre idiomas en este corpus, aunque no genera rankings idénticos. Para comprobarla usaría un conjunto bilingüe mayor, con traducciones revisadas, mantendría corpus y parámetros fijos y mediría recuperación de evidencia correcta y solapamiento de rankings. Estas tres observaciones no demuestran por sí solas una propiedad universal del modelo.

## B5 · Escribe tu propio quiz con ground truth

Se verificaron las secciones originales y se añadieron bloques `<details>` al notebook. La tercera pregunta exige combinar ambos artículos.

| Pregunta nueva | Ground truth verificado | Sección | Respuesta real | Fuentes | Evaluación |
|---|---|---|---|---|---|
| How many identical layers are used in the encoder stack of the base Transformer? | The encoder contains N=6 identical layers. | Attention §3.1 Encoder and Decoder Stacks | The encoder stack of the base Transformer uses N = 6 identical layers. (Attention Is All You Need) | attention_paper.pdf#9; attention_paper.pdf#31; attention_paper.pdf#10; attention_paper.pdf#15; attention_paper.pdf#14 | Sí: seis capas. |
| What non-parametric knowledge source is used by RAG, and how are Wikipedia articles split into passages? | The non-parametric memory uses the December 2018 English Wikipedia dump; articles are split into disjoint 100-word passages, yielding about 21 million documents. | RAG §3 Experiments, primer párrafo | RAG uses a dense vector index of Wikipedia as its non-parametric memory. Each Wikipedia article is split into disjoint 100-word chunks, totaling 21M documents. [Retrieval-Augmented Generation \| rag_paper.pdf] | rag_paper.pdf#18; rag_paper.pdf#8; rag_paper.pdf#7; rag_paper.pdf#1; rag_paper.pdf#19 | Correcta en Wikipedia, 100 palabras y 21M; omitió la fecha del dump de la referencia. |
| Both papers use a Transformer-based sequence-to-sequence generator. What does Attention remove from its architecture, and what external memory does RAG add to its generator? | Attention removes recurrence and convolutions, relying on attention. RAG uses BART-large as its seq2seq generator and augments it with a dense vector index of Wikipedia passages accessed by a retriever. | Attention Abstract/§3; RAG §2.2 Retriever: DPR y §2.3 Generator: BART | Attention removes recurrence from its architecture. RAG adds a dense vector index of Wikipedia, accessed with a pre-trained neural retriever (Dense Passage Retriever), as external memory to its generator. (rag_paper.pdf) | rag_paper.pdf#10; rag_paper.pdf#8; rag_paper.pdf#11; rag_paper.pdf#7; rag_paper.pdf#12 | Parcial: omite las convoluciones y solo recupera RAG, sin evidencia de Attention. |

Referencias primarias: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762), §3.1 y Abstract/§3; [Retrieval-Augmented Generation](https://arxiv.org/pdf/2005.11401), §3 (primer párrafo), §2.2 y §2.3. La sección del generador es **2.3**, y la del retriever es **2.2**; se corrigieron esas referencias en la exportación final.

---

# Reflexión

## R1 · Reflexión final con tus datos

### 1. Contexto insuficiente

En A2, el sistema no inventó ni el NIT de Café Nube ni una tasa de interés ausente del corpus. En B3, las tres preguntas trampa recibieron respuestas que reconocían falta de evidencia. Sin embargo, los scores de A4 para preguntas sin respuesta llegaron a 0.397 y 0.437, superiores a algunos scores de preguntas respondibles; la similitud temática no reemplaza la comprobación de que el contexto contiene el dato.

### 2. Fuentes de la bonus

La recuperación de la bonus devolvió cinco fragmentos de RAG, con índices 10, 12, 13, 86 y 11. No apareció Attention entre las fuentes, aunque la respuesta identificó BART y su relación con Transformer. Por eso una respuesta aparentemente correcta puede esconder cobertura incompleta: añadiría evaluación explícita de evidencia de ambos papers para preguntas cruzadas.

### 3. Parámetros de recuperación

Las nueve configuraciones de B2 lograron 5/5, por lo que no se observó una mejora en aciertos al aumentar k o el tamaño de chunk. Sí cambiaron los recursos: el índice pasó de 326 a 96 chunks entre tamaños 400 y 1500, y k=3 recuperó 12.604 caracteres frente a 32.908 con k=8 para las mismas cinco preguntas. Mantendría el overlap relacionado con el tamaño para conservar continuidad, y validaría la configuración candidata 900/120 con k=3 en más preguntas antes de adoptarla.

### 4. Separación de embeddings y generación

El modelo de embeddings transforma textos y consultas en vectores para buscar evidencia, mientras que el generativo redacta una respuesta a partir de los fragmentos recuperados. En esta ejecución se cambió el modelo de embeddings por indicación de la estudiante y fue necesario reconstruir los índices; cambiar únicamente el prompt de A3 permitió reutilizar el índice sin recalcular esos vectores. Separar ambos componentes permite evaluar si un fallo proviene de la recuperación o de la generación y controlar mejor sus costos.

### 5. Paso a producción

El prototipo usa Qdrant en memoria, por lo que falta persistencia con respaldo y una política de actualización del corpus. Agregaría autenticación y filtros de acceso por documento antes de recuperar contenido, además de guardar las claves en un gestor de secretos. Mantendría un conjunto de evaluación con preguntas respondibles, trampa y cruzadas, ya que el 80% de hit de C4 no refleja por completo la calidad de la respuesta final. También registraría fuentes, scores, abstenciones, errores y latencias sin secretos, e instrumentaría tokens, cuotas y cachés para observar costos y evitar reconstrucciones innecesarias.

---

# Retos elegidos

Se hicieron C3, C4 y C5: 35 puntos disponibles, sujetos al **tope de 30**. C1, C2 y C6 no se eligieron; son opcionales según la página.

## C3 · App Gradio para los papers con filtro por paper

Evidencia: [captura de la app](evidencias/C3_gradio.png), código y comparación en el notebook, y `C3` en el JSON de resultados. El Dropdown contiene Todos, Attention Is All You Need y Retrieval-Augmented Generation. El filtro usa `Filter` y `FieldCondition(key="metadata.source", ...)` de Qdrant; se aplica durante la recuperación, no después de generar.

| Filtro | Misma pregunta | Respuesta | Fuentes y scores |
|---|---|---|---|
| Todos | How many attention heads does the base Transformer model use? | The base Transformer model uses 8 attention heads. This is stated in the section describing multi-head attention: "In this work we employ h = 8 parallel attention layers, or heads," and confirmed in Table 3 where the base model lists h = 8. [Attention Is All You Need \| attention_paper.pdf] | attention_paper.pdf#14: 0.4225; attention_paper.pdf#31: 0.3875; attention_paper.pdf#40: 0.3743; attention_paper.pdf#9: 0.3740; attention_paper.pdf#39: 0.3691 |
| Retrieval-Augmented Generation | How many attention heads does the base Transformer model use? | The provided context does not contain any excerpts from academic papers, so there is no evidence to answer the question. | rag_paper.pdf#82: 0.2157; rag_paper.pdf#14: 0.2154; rag_paper.pdf#86: 0.1929; rag_paper.pdf#13: 0.1828; rag_paper.pdf#16: 0.1781 |

Con Todos se recuperó Attention y se respondió ocho cabezas. Al limitar a RAG, los candidatos fueron de ese paper y no superaron el umbral para proporcionar contexto útil; el sistema se abstuvo. La app muestra también los candidatos con sus scores para distinguir encontrar vecinos de tener evidencia suficiente para responder.

## C4 · Evaluación automática de la recuperación

Evidencias: [10 preguntas CSV](eval/preguntas.csv), función `hit_at_k(retriever, filas)` en el notebook y tabla siguiente. La función exige simultáneamente la fuente esperada y la palabra clave dentro de un fragmento; normaliza mayúsculas y espacios, y devuelve un porcentaje entre 0 y 100.

| chunk_size | hit@3 (%) | hit@5 (%) |
|---|---|---|
| 400 | 80 | 80 |
| 900 | 80 | 80 |
| 1500 | 80 | 80 |

En los seis casos se acertaron 8 de 10 consultas: la métrica no distinguió estas configuraciones. Primera limitación: una paráfrasis o un cambio de extracción puede expresar la respuesta correcta sin contener la palabra clave, causando un falso negativo. Segunda limitación: una palabra como «64» puede aparecer en el paper correcto sin ser evidencia de la respuesta pedida, causando un falso positivo; tampoco se evalúa si el generador usa bien esa evidencia.

## C5 · MMR frente a similitud pura

Se comparó similitud con `MMR(k=5, fetch_k=20, lambda_mult=0.5)` usando el mismo índice. El JSON y la salida del notebook imprimen el contenido íntegro de todos los chunks de cada lado.

| Pregunta | Similitud: fragmentos | MMR: fragmentos | Redundancia y cobertura |
|---|---|---|---|
| What BLEU score did the Transformer (big model) achieve on WMT 2014 English-to-German? | attention_paper.pdf#28; attention_paper.pdf#1; attention_paper.pdf#26; attention_paper.pdf#31; attention_paper.pdf#24 | attention_paper.pdf#28; attention_paper.pdf#0; attention_paper.pdf#1; attention_paper.pdf#31; attention_paper.pdf#35 | MMR reduce repetición temática, pero introduce el chunk de portada/autores (0) y omite el chunk 26 de la tabla de resultados. |
| What are the two RAG model variants proposed in the paper and how do they differ? | rag_paper.pdf#10; rag_paper.pdf#11; rag_paper.pdf#24; rag_paper.pdf#12; rag_paper.pdf#91 | rag_paper.pdf#10; rag_paper.pdf#40; rag_paper.pdf#52; rag_paper.pdf#24; rag_paper.pdf#30 | MMR amplía temas (tablas e impacto), pero desplaza fragmentos 11 y 12 que explican las variantes; mayor diversidad no garantiza mejor evidencia. |
| The RAG paper uses FAISS for fast retrieval with an HNSW approximation. The Attention paper proposes the Transformer architecture. What seq2seq model does RAG use as its generator component, and is it based on the Transformer? | rag_paper.pdf#10; rag_paper.pdf#12; rag_paper.pdf#13; rag_paper.pdf#86; rag_paper.pdf#11 | rag_paper.pdf#10; attention_paper.pdf#0; rag_paper.pdf#40; rag_paper.pdf#86; rag_paper.pdf#25 | MMR incluye ambos papers, mientras similitud solo devuelve RAG; el chunk 0 de Attention empieza con portada, permisos y autores, por lo que la cobertura por título no prueba evidencia útil. |

MMR puede ayudar más en preguntas amplias o cruzadas, cuando se necesita evidencia de varios aspectos y el ranking de similitud resulta concentrado en uno solo. En preguntas puntuales, como BLEU o la diferencia exacta entre variantes, puede empeorar al sustituir fragmentos precisos por otros más diversos pero menos útiles. En estas pruebas no se midió una mejora de respuestas generadas con MMR: se comparó la recuperación, que es lo que pide el reto.
