"""Cola de jobs y registro de estado en Redis.

Patrón del módulo 7: la API guarda el estado `pending` y recién después encola el id
(`rpush`), para que el cliente nunca reciba un 404 por consultar antes de que exista. El
worker saca ids con `blpop` y actualiza el estado a medida que avanza.

Además, cada conversación (`thread_id`) admite un solo job activo a la vez: dos turnos
concurrentes sobre el mismo hilo pisarían el checkpoint.
"""

from redis.asyncio import Redis

from app.config import Settings
from app.schemas import JobRecord, utcnow

_JOB_PREFIX = "job:"
_THREAD_LOCK_PREFIX = "thread_lock:"
_RESUME_GUARD_PREFIX = "job_resume:"


class ThreadBusyError(Exception):
    def __init__(self, thread_id: str, job_id: str | None) -> None:
        super().__init__(f"El thread {thread_id} ya tiene un job activo ({job_id})")
        self.thread_id = thread_id
        self.job_id = job_id


class JobStore:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._queue = settings.queue_name
        self._ttl = settings.job_ttl_seconds

    @property
    def redis(self) -> Redis:
        return self._redis

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self._redis.get(f"{_JOB_PREFIX}{job_id}")
        return JobRecord.model_validate_json(raw) if raw is not None else None

    async def save(self, job: JobRecord) -> None:
        job.updated_at = utcnow()
        await self._redis.set(f"{_JOB_PREFIX}{job.job_id}", job.model_dump_json(), ex=self._ttl)

    async def enqueue(self, job: JobRecord) -> None:
        """Primero el estado, después la cola (evita el 404 por race condition)."""
        await self.save(job)
        await self._redis.rpush(self._queue, job.job_id)

    async def dequeue(self, timeout: int = 5) -> str | None:
        """Pop bloqueante: espera sin consumir CPU. Devuelve None si vence el timeout."""
        item = await self._redis.blpop([self._queue], timeout=timeout)
        return item[1] if item else None

    async def acquire_thread(self, thread_id: str, job_id: str) -> None:
        ok = await self._redis.set(
            f"{_THREAD_LOCK_PREFIX}{thread_id}", job_id, nx=True, ex=self._ttl
        )
        if not ok:
            holder = await self._redis.get(f"{_THREAD_LOCK_PREFIX}{thread_id}")
            raise ThreadBusyError(thread_id, holder)

    async def release_thread(self, thread_id: str, job_id: str) -> None:
        key = f"{_THREAD_LOCK_PREFIX}{thread_id}"
        if await self._redis.get(key) == job_id:
            await self._redis.delete(key)

    async def claim_resume(self, job_id: str) -> bool:
        """Garantiza que un job pausado se reanude una sola vez aunque lleguen dos /approve."""
        return bool(
            await self._redis.set(f"{_RESUME_GUARD_PREFIX}{job_id}", "1", nx=True, ex=self._ttl)
        )
