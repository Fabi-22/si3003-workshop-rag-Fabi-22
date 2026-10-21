# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** Fabiola Valencia Barrios
- **Código estudiantil:** 1000352691
- **GitHub:** Fabi-22
- **Correo institucional:** fvalenciab@eafit.edu.co
- **Fecha de ejecución:** 21 de octubre de 2026
- **Estado:** trabajo y evidencias completos.

## Entrega y evidencias

[Respuestas, tablas y reflexión](RESPUESTAS.md) · [Paso a paso](PASO_A_PASO.md) · [Incidencias resueltas](INCIDENCIAS.md)

Los notebooks incluyen código, conteos de ejecución y salidas reales de Colab. La simulación T1 se ejecutó localmente. Los datos completos están en [resultados A](evidencias/resultados_A.json) y [resultados B](evidencias/resultados_B.json); los resultados parciales o negativos se conservan y analizan.

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Nodos por capa | 2 | ✅ | [Ver](notebooks/01_teoria_hnsw.ipynb) |
| T2 | ef y k | 2 | ✅ | [Ver](RESPUESTAS.md#t2--ef-k-y-el-tamaño-de-la-cola) |
| T3 | Inserción y búsqueda | 2 | ✅ | [Ver](RESPUESTAS.md#t3--inserción-frente-a-búsqueda) |
| T4 | Invariante de FAISSVectorStore | 2 | ✅ | [Ver](RESPUESTAS.md#t4--el-invariante-de-faissvectorstore) |
| T5 | Costo de fuerza bruta | 2 | ✅ | [Ver](RESPUESTAS.md#t5--el-costo-de-la-fuerza-bruta) |
| A1 | App RAG completa y cinco preguntas | 4 | ✅ | [Ver](evidencias/A1_gradio.png) |
| A2 | Corpus propio de ocho documentos | 6 | ✅ | [Ver](data/company_documents.json) |
| A3 | Prompt antes y después | 5 | ✅ | [Ver](evidencias/resultados_A.json) |
| A4 | Fuentes con scores | 5 | ✅ | [Ver](evidencias/A4_scores.png) |
| B1 | Quiz Q1–Q5 y bonus | 6 | ✅ | [Ver](evidencias/resultados_B.json) |
| B2 | Nueve configuraciones experimentales | 8 | ✅ | [Ver](evidencias/resultados_B.json) |
| B3 | Tres preguntas trampa | 6 | ✅ | [Ver](evidencias/resultados_B.json) |
| B4 | Comparación EN/ES | 5 | ✅ | [Ver](evidencias/resultados_B.json) |
| B5 | Tres preguntas propias con ground truth | 5 | ✅ | [Ver](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| R1 | Reflexión sustentada en datos | 10 | ✅ | [Ver](RESPUESTAS.md) |
| C3 | Gradio con filtro por paper | 15 | ✅ | [Ver](evidencias/C3_gradio.png) |
| C4 | Evaluación automática de recuperación | 12 | ✅ | [Ver](eval/preguntas.csv) |
| C5 | MMR frente a similitud | 8 | ✅ | [Ver](evidencias/resultados_B.json) |

**Autoevaluación solicitada: 100/100** = 70 del núcleo + min(35 de C3/C4/C5, 30). Es una estimación de cumplimiento, sujeta a la revisión del profesor; no representa una nota concedida. Los retos son opcionales con tope de 30; C1, C2 y C6 no se seleccionaron ni se reclaman.

## Modelos utilizados

```python
GENERATION_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
```

El embedding produjo vectores de 2048 dimensiones. Se desactivó thinking con `chat_template_kwargs={"enable_thinking": False}` para obtener respuestas completas dentro del presupuesto de tokens; se añadieron reintentos acotados para errores transitorios del servicio. No se incluye ninguna API key.

## Cómo ejecutar

1. Abre [02b](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) y [03](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) en Colab y guarda una copia personal.
2. Guarda `NVIDIA_API_KEY` en Secrets y habilita acceso para cada notebook. Ejecuta instalación y configuración.
3. En 02b, la primera app corresponde a la demostración A1. Su celda `launch(debug=True)` queda activa: interrúmpela desde **Entorno de ejecución → Interrumpir la ejecución** para continuar con los experimentos añadidos al final. No ejecutes la celda opcional `gr.close_all()` después de lanzar la app que quieras conservar.
4. Las últimas tres celdas de 02b ejecutan A2–A4 sobre Café Nube, repiten A1 con los modelos finales y restauran el corpus propio para la app con scores. El corpus está incluido en la celda experimental y también en `data/company_documents.json`; no depende de archivos privados de Drive.
5. En 03 ejecuta las partes 0–5 y luego las celdas añadidas de experimentos, Gradio y exportación. Allí están B1–B5 y C3/C4/C5. Se construyen tres índices (400/900/1500) y se reutilizan en los barridos de k y umbral.
6. Las URLs `gradio.live` son temporales y dependen de la sesión de Colab. Las capturas y salidas guardadas permiten revisar la entrega después de que caduquen; al ejecutar se crean nuevas URLs.

## Decisiones y aprendizajes

El corpus ficticio de Café Nube tiene ocho documentos breves y ocho chunks de 400 caracteres con overlap 40. En A3 el prompt final incorporó citas, abstención fija y un máximo de tres oraciones. Un score alto no garantiza que un documento responda la pregunta: las consultas ajenas al corpus también recuperaron candidatos con scores altos. En B2 las nueve configuraciones lograron 5/5 en este quiz pequeño, por lo que no se puede afirmar una mejora de exactitud. k=3 redujo el contexto frente a k=5 sin perder aciertos en esas cinco consultas. En C4 hit@3 y hit@5 fueron 80% para los tres tamaños, con las limitaciones de una comprobación léxica. MMR diversificó los resultados, pero también introdujo una portada y desplazó evidencia útil; diversidad no equivale a mejor respuesta. La pregunta propia que cruza ambos papers quedó parcialmente respondida y se explica en B5.

## Archivos adicionales

`04_challenges_starter.ipynb` se conserva como plantilla original del curso; no es evidencia ejecutada ni se reclama su contenido. Los retos seleccionados se implementaron y ejecutaron en 03; [04_challenges.ipynb](notebooks/04_challenges.ipynb) conserva una copia completa del entorno y esas mismas salidas para facilitar su ejecución independiente. No son mediciones adicionales. El correo de entrega debe enviarse a **jipadillab@eafit.edu.co** desde el correo institucional, con el asunto y contenido preparados en el borrador de entrega.
