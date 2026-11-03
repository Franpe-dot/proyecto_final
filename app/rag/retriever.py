"""Recuperación sobre la base de políticas de crédito de FinCrece S.A.

ChromaDB corre como servidor y se consulta con `AsyncHttpClient`, así que el acceso a la
base es asíncrono de punta a punta. Los embeddings (`intfloat/multilingual-e5-*`) se
calculan localmente: son trabajo de CPU, no de I/O, y se corren en un hilo aparte con
`asyncio.to_thread` para no bloquear el event loop.
"""

import asyncio
import logging
from typing import Any, Literal

import chromadb
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.config import Settings

logger = logging.getLogger(__name__)

# Temas = documentos de `data/`, sin el prefijo numérico. Se guardan como metadata en la
# ingesta y permiten filtrar la búsqueda por documento.
Tema = Literal["otorgamiento", "tasas_y_costos", "mora_y_cobranzas", "refinanciacion_cancelacion"]


class SearchQuery(BaseModel):
    consulta: str = Field(
        min_length=3,
        max_length=500,
        description="Qué se quiere saber, en lenguaje natural. Ej.: 'TNA para calificación B'.",
    )
    k: int = Field(default=3, ge=1, le=6, description="Cantidad de fragmentos a devolver.")
    tema: Tema | None = Field(
        default=None, description="Documento al que restringir la búsqueda (opcional)."
    )


class Fragment(BaseModel):
    fuente: str
    seccion: str
    relevancia: float = Field(ge=-1, le=1)
    texto: str


class SearchResult(BaseModel):
    consulta: str
    tema: Tema | None
    fragmentos: list[Fragment]


class PolicyRetriever:
    """Cliente de la colección de políticas. Conecta y carga el modelo la primera vez que se usa."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncClientAPI | None = None
        self._collection: AsyncCollection | None = None
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    async def collection(self) -> AsyncCollection:
        async with self._lock:
            if self._collection is None:
                self._client = await chromadb.AsyncHttpClient(
                    host=self._settings.chroma_host, port=self._settings.chroma_port
                )
                self._collection = await self._client.get_or_create_collection(
                    self._settings.chroma_collection,
                    # Chroma usa L2 por defecto; e5 espera similitud coseno.
                    metadata={"hnsw:space": "cosine"},
                    # Los vectores los calculamos nosotros: Chroma no embebe nada.
                    embedding_function=None,
                )
            return self._collection

    async def _get_model(self) -> SentenceTransformer:
        async with self._lock:
            if self._model is None:
                logger.info("Cargando modelo de embeddings %s", self._settings.embedding_model)
                self._model = await asyncio.to_thread(
                    SentenceTransformer, self._settings.embedding_model
                )
            return self._model

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        model = await self._get_model()
        vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        # La familia e5 exige prefijos asimétricos para documentos y consultas.
        return await self._embed([f"passage: {t}" for t in texts])

    async def search(self, query: SearchQuery) -> SearchResult:
        collection = await self.collection()
        [vector] = await self._embed([f"query: {query.consulta}"])
        where: dict[str, Any] | None = {"tema": query.tema} if query.tema else None
        raw = await collection.query(
            query_embeddings=[vector],
            n_results=query.k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = (raw["documents"] or [[]])[0]
        metadatas = (raw["metadatas"] or [[]])[0]
        distances = (raw["distances"] or [[]])[0]
        fragments = [
            Fragment(
                fuente=str(meta["source"]),
                seccion=str(meta["section"]),
                # Distancia coseno → similitud.
                relevancia=round(1 - distance, 3),
                texto=doc,
            )
            for doc, meta, distance in zip(documents, metadatas, distances, strict=True)
        ]
        return SearchResult(consulta=query.consulta, tema=query.tema, fragmentos=fragments)
