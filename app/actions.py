"""Registro de solicitudes de crédito: la acción con efecto real que protege el HITL."""

from redis.asyncio import Redis

from app.schemas import SolicitudRegistrada

_KEY_PREFIX = "solicitud:"
_INDEX_KEY = "solicitudes"


class SolicitudStore:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def register(self, record: SolicitudRegistrada) -> bool:
        """Guarda la solicitud. Idempotente: si el id ya existe no hace nada y devuelve False.

        El id es determinista (thread + turno), así que si el nodo se re-ejecuta después de
        un reinicio no se registra dos veces la misma solicitud.
        """
        created = await self._redis.set(
            f"{_KEY_PREFIX}{record.solicitud_id}", record.model_dump_json(), nx=True
        )
        if created:
            await self._redis.rpush(_INDEX_KEY, record.solicitud_id)
        return bool(created)

    async def list(self, limit: int = 50) -> list[SolicitudRegistrada]:
        ids = await self._redis.lrange(_INDEX_KEY, -limit, -1)
        if not ids:
            return []
        raw = await self._redis.mget([f"{_KEY_PREFIX}{i}" for i in ids])
        return [SolicitudRegistrada.model_validate_json(r) for r in raw if r is not None]
