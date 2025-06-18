## RAG Api framework
Este repositorio contiene un framework para crear APIs de Recuperación de Información (RAG) utilizando
Python y FastAPI. El framework está diseñado para ser modular, escalable y fácil de usar.

### Instalación
El entorno de desarrollo utiliza Docker y Docker Compose, con las librerías instaladas a partir de poetry.

#### Requisitos previos
- Docker

### Iniciar contenedor Docker con Docker Compose

```bash
docker-compose up -d
```

### Composición del proyecto
- `app/`: Contiene el código fuente de la aplicación.
    - `app/main.py`: Punto de entrada de la aplicación FastAPI.
    - `app/chroma_client.py`: Cliente para interactuar con ChromaDB.
- `app/embedding/` : Módulo para manejar embeddings.
    - `app/embedding/generator.py`: Generador de embeddings.
    - `app/embedding/collection_search.py`: Implementación de búsqueda en colecciones.

### Documentación API
