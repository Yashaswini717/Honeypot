from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from config.logging_config import LoggerMixin
from config.settings import settings
from core.exceptions import DatabaseError
from core.utils import generate_unique_id

from .models import Base, HoneytokenCreate, HoneytokenDB, HoneytokenResponse, utcnow_naive


class HoneytokenStore(LoggerMixin):
    """Store and track honeytokens."""

    @staticmethod
    def _create_engine(database_url: str):
        """Create a database engine with sane defaults for SQLite files."""
        engine_kwargs = {"echo": settings.database_echo}

        if database_url.startswith("sqlite:///"):
            db_path = database_url.removeprefix("sqlite:///")
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                engine_kwargs["poolclass"] = NullPool

        return create_engine(database_url, **engine_kwargs)

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize honeytoken store.

        Args:
            database_url: Database connection URL (uses settings if not provided)
        """
        self.database_url = database_url or settings.database_url
        self.engine = self._create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        self.logger.info("honeytoken_store_initialized", database_url=self.database_url)

    def create_honeytoken(self, honeytoken: HoneytokenCreate) -> HoneytokenResponse:
        """
        Create and store a honeytoken.

        Args:
            honeytoken: Honeytoken data

        Returns:
            Created honeytoken with ID

        Raises:
            DatabaseError: If creation fails
        """
        try:
            with self.SessionLocal() as session:
                db_token = HoneytokenDB(
                    token_id=generate_unique_id(),
                    token_type=honeytoken.token_type,
                    token_value=honeytoken.token_value,
                    honeypot_id=honeytoken.honeypot_id,
                    file_path=honeytoken.file_path,
                    token_metadata=honeytoken.token_metadata,
                )
                session.add(db_token)
                session.commit()
                session.refresh(db_token)
                
                self.logger.info(
                    "honeytoken_created",
                    token_id=db_token.token_id,
                    token_type=db_token.token_type,
                )
                
                return HoneytokenResponse.model_validate(db_token)
        except Exception as e:
            self.logger.error("honeytoken_creation_failed", error=str(e))
            raise DatabaseError(f"Failed to create honeytoken: {e}") from e

    def get_honeytoken(self, token_id: str) -> Optional[HoneytokenResponse]:
        """
        Get honeytoken by ID.

        Args:
            token_id: Token ID

        Returns:
            Honeytoken if found, None otherwise
        """
        try:
            with self.SessionLocal() as session:
                stmt = select(HoneytokenDB).where(HoneytokenDB.token_id == token_id)
                result = session.execute(stmt).scalar_one_or_none()
                
                if result:
                    return HoneytokenResponse.model_validate(result)
                return None
        except Exception as e:
            self.logger.error("honeytoken_retrieval_failed", token_id=token_id, error=str(e))
            raise DatabaseError(f"Failed to get honeytoken: {e}") from e

    def check_honeytoken(self, token_value: str) -> Optional[HoneytokenResponse]:
        """
        Check if a token value matches any honeytoken.

        Args:
            token_value: Token value to check

        Returns:
            Matching honeytoken if found
        """
        try:
            with self.SessionLocal() as session:
                stmt = select(HoneytokenDB).where(
                    HoneytokenDB.token_value == token_value,
                    HoneytokenDB.is_active == True,
                )
                result = session.execute(stmt).scalar_one_or_none()
                
                if result:
                    # Log access.
                    #
                    # This was datetime.now() — local time — while every other
                    # timestamp in this schema comes from the database's own
                    # func.now(), which is UTC. RewardCalculator compares this
                    # field against a decision's selected_at, so on a non-UTC
                    # host the strongest signal the bandit has was mis-ordered
                    # by the size of the offset: ahead of UTC (as here, +5:30)
                    # an access from hours ago still satisfies the comparison
                    # and pays reward 1.0 to an arm that earned nothing.
                    result.accessed_at = utcnow_naive()
                    result.access_count += 1
                    session.commit()
                    
                    self.logger.warning(
                        "honeytoken_accessed",
                        token_id=result.token_id,
                        token_type=result.token_type,
                        access_count=result.access_count,
                    )
                    
                    return HoneytokenResponse.model_validate(result)
                return None
        except Exception as e:
            self.logger.error("honeytoken_check_failed", error=str(e))
            raise DatabaseError(f"Failed to check honeytoken: {e}") from e

    def list_honeytokens(
        self,
        honeypot_id: Optional[str] = None,
        token_type: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[HoneytokenResponse]:
        """
        List honeytokens with filters.

        Args:
            honeypot_id: Filter by honeypot ID
            token_type: Filter by token type
            active_only: Only return active tokens
            limit: Maximum number to return

        Returns:
            List of honeytokens
        """
        try:
            with self.SessionLocal() as session:
                stmt = select(HoneytokenDB)
                
                if honeypot_id:
                    stmt = stmt.where(HoneytokenDB.honeypot_id == honeypot_id)
                if token_type:
                    stmt = stmt.where(HoneytokenDB.token_type == token_type)
                if active_only:
                    stmt = stmt.where(HoneytokenDB.is_active == True)
                
                stmt = stmt.order_by(HoneytokenDB.created_at.desc()).limit(limit)
                
                results = session.execute(stmt).scalars().all()
                return [HoneytokenResponse.model_validate(r) for r in results]
        except Exception as e:
            self.logger.error("honeytoken_list_failed", error=str(e))
            raise DatabaseError(f"Failed to list honeytokens: {e}") from e

    def deactivate_honeytoken(self, token_id: str) -> bool:
        """
        Deactivate a honeytoken.

        Args:
            token_id: Token ID

        Returns:
            True if deactivated, False if not found
        """
        try:
            with self.SessionLocal() as session:
                stmt = select(HoneytokenDB).where(HoneytokenDB.token_id == token_id)
                result = session.execute(stmt).scalar_one_or_none()
                
                if result:
                    result.is_active = False
                    session.commit()
                    self.logger.info("honeytoken_deactivated", token_id=token_id)
                    return True
                return False
        except Exception as e:
            self.logger.error("honeytoken_deactivation_failed", token_id=token_id, error=str(e))
            raise DatabaseError(f"Failed to deactivate honeytoken: {e}") from e

    def close(self) -> None:
        """Dispose engine resources."""
        self.engine.dispose()
