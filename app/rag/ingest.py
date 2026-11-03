"""Ingesta de `data/*.md` en ChromaDB: un fragmento por sección `##`, con metadata.

    python -m app.rag.ingest            # indexa solo si la colección está vacía
    python -m app.rag.ingest --rebuild  # vuelve a indexar todo (upsert)

El worker llama a `ensure_index` al arrancar, así que en Docker no hace falta correrlo.
"""

import argparse
import asyncio
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from app.config import DATA_DIR, get_settings
from app.rag.retriever import PolicyRetriever

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    section: str
    tema: str


def split_by_section(path: Path) -> list[Chunk]:
    """Corta un documento en sus secciones `##`.

    Cada fragmento conserva el título del documento para no perder de qué política habla.
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    title, *sections = re.split(r"^## ", text, flags=re.MULTILINE)
    title = title.strip().lstrip("# ")
    tema = re.sub(r"^\d+_", "", path.stem)
    chunks: list[Chunk] = []
    for i, section in enumerate(sections):
        heading, _, body = section.partition("\n")
        chunks.append(
            Chunk(
                id=f"{path.name}::{i}",
                text=f"{title}\n## {heading.strip()}\n{body.strip()}",
                source=path.name,
                section=heading.strip(),
                tema=tema,
            )
        )
    return chunks


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    chunks = [c for path in sorted(data_dir.glob("*.md")) for c in split_by_section(path)]
    if not chunks:
        raise RuntimeError(f"No se encontraron archivos .md en {data_dir}")
    return chunks


async def ensure_index(retriever: PolicyRetriever, rebuild: bool = False) -> int:
    """Indexa los documentos si la colección está vacía (o siempre, con `rebuild`).

    Usa `upsert` con ids deterministas: correrlo dos veces no duplica fragmentos.
    Devuelve la cantidad de fragmentos en la colección.
    """
    collection = await retriever.collection()
    count = await collection.count()
    if count and not rebuild:
        logger.info("Colección con %d fragmentos: no se reindexa", count)
        return count
    chunks = load_chunks()
    embeddings = await retriever.embed_passages([c.text for c in chunks])
    await collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,  # type: ignore[arg-type]
        metadatas=[{"source": c.source, "section": c.section, "tema": c.tema} for c in chunks],
    )
    logger.info("Indexados %d fragmentos de %s", len(chunks), DATA_DIR)
    return await collection.count()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Indexa data/ en ChromaDB")
    parser.add_argument("--rebuild", action="store_true", help="Reindexa aunque ya haya datos")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    total = await ensure_index(PolicyRetriever(get_settings()), rebuild=args.rebuild)
    print(f"Colección lista: {total} fragmentos")


if __name__ == "__main__":
    asyncio.run(_main())
