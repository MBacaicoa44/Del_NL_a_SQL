## Trabajo Fin de Grado – Del Lenguaje Natural a las consultas SQL: Cómo los modelos de lenguaje aprenden a hablar con las bases de datos

Este proyecto corresponde a mi Trabajo Fin de Grado (TFG), centrado en el desarrollo y evaluación de un sistema de traducción automática de preguntas en lenguaje natural a consultas SQL, utilizando modelos de lenguaje preentrenados.


### Metodología
- Se utiliza el dataset **Spider**
- Las consultas se clasifican automáticamente por niveles de dificultad estructural.
- Las preguntas se analizan lingüísticamente con **spaCy**, y se clasifican por niveles gramaticales.
- Se entrena y evalúa cada modelo y se calcula la métrica **BLEU** para comparar la calidad de las traducciones generadas.
- Se analizan los resultados con visualizaciones (barras, boxplots, líneas y heatmaps).

### Contenido del repositorio
- `data/`: datos raw de Spider.
- `docs/`: memoria del trabajo final de grado.
- `finetuned_models/`: modelos fine-tuneados resultados del notebook `01_train_models.ipynb`
- `notebooks`: notebooks utilizados para entrenar los modelos (`01_train_models.ipynb`) y para visualizar los resultados (02_results_analysis.ipynb).
- `predictions`: ficheros csv con los resultados finales de los datos utilizados como testing. (consulta generada, BLEU y clasificación)
- `procesed`: datasets procesados.
- `src/`: código fuente para preprocesamiento, clasificación e inferencia.

