"""Progresso no terminal, sem prompts, respostas ou credenciais."""

from contextlib import contextmanager
from datetime import datetime
from threading import Event, Thread
from time import monotonic


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


@contextmanager
def activity(label, *, interval=15):
    """Heartbeat para etapas bloqueantes; termina tambem em caso de erro."""
    started = monotonic()
    done = Event()

    def heartbeat():
        while not done.wait(interval):
            log(f"{label}: em andamento ({monotonic() - started:.0f}s)")

    log(label)
    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        log(f"{label}: interrompido apos {monotonic() - started:.1f}s")
        raise
    else:
        log(f"{label}: concluido em {monotonic() - started:.1f}s")
    finally:
        done.set()
        worker.join()
