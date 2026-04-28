"""Triple-path authentication: API key, Gateway-forwarded user, or direct JWT.

Usage in endpoints:
    from app.api.auth import require_auth, CallerContext, authorize_agent_access

    @router.post("/reviews")
    async def start_review(
        ...,
        caller: CallerContext = Depends(require_auth),
    ):
        authorize_agent_access(caller, agent)
"""

import base64
import hmac
import logging
from dataclasses import dataclass
from typing import Optional

import jwt
import redis as redis_lib
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database_layer.db_config import get_db
from app.database_layer.db_models import Users, RolesPermissions, Permissions
import app.cache_db.redis_client as redis_client

logger = logging.getLogger("kb_review")

SUPER_ADMIN_PERMISSION = "super_admin_access"


@dataclass
class CallerContext:
    """Authenticated caller for the current request."""

    auth_type: str  # "api_key" | "gateway" | "jwt"
    user_id: Optional[int] = None
    company_id: Optional[int] = None
    is_super_admin: bool = False

    @property
    def is_service(self) -> bool:
        """True if the caller is an internal service (API key auth)."""
        return self.auth_type == "api_key"

    def can_access_agent(self, agent_company_id: Optional[int]) -> bool:
        """Check if this caller can access an agent with the given company_id."""
        if self.is_service or self.is_super_admin:
            return True
        if agent_company_id is None:
            return False
        return self.company_id == agent_company_id


def authorize_agent_access(caller: CallerContext, agent) -> None:
    """Raise 404 if caller cannot access this agent.

    Returns 404 (not 403) to avoid leaking that the agent exists.
    """
    if not caller.can_access_agent(agent.company_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent.id} not found",
        )


# ── Internal helpers ─────────────────────────────────────────────────────


def _verify_api_key(api_key: str) -> None:
    """Validate API key. Raises 403 on mismatch, 500 if not configured."""
    expected = settings.X_API_KEY.strip()
    if not expected:
        logger.error("X_API_KEY env var not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured on server",
        )
    if not hmac.compare_digest(api_key.strip(), expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


def _decode_jwt(token: str) -> dict:
    """Decode and validate a JWT token using the shared HMAC-SHA256 secret.

    Uses the same algorithm and Base64-encoded secret as the Java API Gateway
    (AuthenticationFilter.java) and Auth Service (JwtServiceImpl.java).
    PyJWT automatically validates exp.
    """
    secret_b64 = settings.JWT_SECRET
    if not secret_b64:
        logger.error("JWT_SECRET env var not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret not configured on server",
        )
    try:
        secret_bytes = base64.b64decode(secret_b64)
        return jwt.decode(
            token,
            secret_bytes,
            algorithms=["HS256"],
            options={"require": ["exp"]},  # "id" is a custom claim — validated after decode
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


def _check_super_admin(db: Session, role_id: int) -> bool:
    """Check if a role has the super_admin_access permission."""
    result = (
        db.query(RolesPermissions)
        .join(Permissions, RolesPermissions.permissions_id == Permissions.id)
        .filter(
            RolesPermissions.roles_id == role_id,
            Permissions.name == SUPER_ADMIN_PERMISSION,
        )
        .first()
    )
    return result is not None


def _build_user_caller(user_id: int, auth_type: str, db: Session) -> CallerContext:
    """Build a CallerContext from a user ID + DB lookup.

    Shared by both the Gateway (X-User-Id) and direct JWT paths.
    """
    user = (
        db.query(Users)
        .filter(Users.id == user_id, Users.status == "ACTIVE")
        .first()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return CallerContext(
        auth_type=auth_type,
        user_id=user_id,
        company_id=user.company_id,
        is_super_admin=_check_super_admin(db, user.role_id),
    )


def _check_token_blacklist(token: str, user_id: int, iat: Optional[int]) -> None:
    """Check Redis for token blacklist and session invalidation.

    Only called on the JWT auth path. API key and Gateway paths skip this.
    Behavior when Redis is not configured depends on APP_ENV:
    - production: fail-closed (503)
    - dev/other: warn and allow through
    When Redis is configured but unreachable: always fail-closed (503).
    """
    if not redis_client.is_redis_configured():
        if settings.APP_ENV.lower() == "production":
            logger.error("Redis not configured in production — rejecting direct JWT")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Token validation service unavailable",
            )
        logger.warning("Redis not configured — skipping JWT blacklist check")
        return

    try:
        if redis_client.is_token_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked. Please login again.",
            )
        if redis_client.is_token_invalidated(user_id, iat):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired due to security update. Please login again.",
            )
    except redis_client.RedisCheckError:
        logger.error("Redis unavailable — rejecting direct JWT (fail-closed)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token validation service unavailable",
        )
    except redis_lib.RedisError as e:
        logger.error(f"Redis error during token check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token validation service unavailable",
        )


# ── FastAPI dependency ───────────────────────────────────────────────────


def require_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> CallerContext:
    """Authenticate the caller via API key, Gateway-forwarded user, or JWT.

    Priority:
    1. X-API-Key + X-User-Id → Gateway-forwarded (company-scoped via user context)
    2. X-API-Key alone → service-to-service (full access)
    3. Authorization: Bearer <jwt> → direct JWT validation (company-scoped)
    4. No credentials → 401

    X-User-Id is ONLY accepted alongside a valid API key. This prevents
    spoofing: since KB Review is standalone (not behind the Gateway), a bare
    X-User-Id could be forged by anyone. Requiring the API key proves the
    caller is a trusted service (the Gateway).
    """
    # Path 1 & 2: API key (required for X-User-Id)
    if x_api_key:
        _verify_api_key(x_api_key)  # raises 403/500 on failure

        # Path 1: Gateway-forwarded user context (API key + X-User-Id)
        if x_user_id:
            try:
                uid = int(x_user_id)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="X-User-Id must be a valid integer",
                )
            return _build_user_caller(uid, auth_type="gateway", db=db)

        # Path 2: Service-to-service (API key alone, full access)
        return CallerContext(auth_type="api_key")

    # Reject bare X-User-Id without API key — prevents spoofing
    if x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-Id requires a valid X-API-Key (Gateway authentication)",
        )

    # Path 3: Direct JWT Bearer token
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header must start with 'Bearer '",
            )
        token = authorization[7:].strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token is empty",
            )
        claims = _decode_jwt(token)
        user_id = claims.get("id")
        if not isinstance(user_id, int):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing or invalid 'id' claim",
            )
        iat = claims.get("iat")
        if iat is not None and not isinstance(iat, (int, float)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has invalid 'iat' claim",
            )
        _check_token_blacklist(token, user_id, int(iat) if iat is not None else None)
        return _build_user_caller(user_id, auth_type="jwt", db=db)

    # Path 4: No credentials
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide X-API-Key, X-User-Id, or Authorization: Bearer <token>",
        headers={"WWW-Authenticate": "Bearer"},
    )
