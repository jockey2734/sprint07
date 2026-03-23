"""SQLAlchemy 2.0 SQLite database engine and session management."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from stock_ai.storage.models_orm import Base


def create_db_engine(db_path: str = "output/stock_ai.db") -> Engine:
    """Create (or open) the SQLite engine and create all tables."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{path.as_posix()}"
    engine = create_engine(url, echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    logger.info(f"Database ready at {path}")
    return engine


@contextmanager
def get_session(engine: Engine) -> Generator[Session, None, None]:
    """Context manager that yields a transactional Session."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
