"""
Main FastAPI application entry point
VOKG API Gateway
"""

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from backend.core.config import settings
from backend.core.logging import setup_logging, get_logger
from backend.database.postgres import init_db, get_db
from backend.database.models import User
from backend.api_gateway.middleware import setup_middleware
from backend.api_gateway.routes import videos, graphs, queries, health
from backend.api_gateway.websocket import websocket_endpoint, get_connection_manager
from backend.api_gateway.auth import get_password_hash, create_access_token, authenticate_user
from backend.api_gateway.dependencies import get_current_active_user
from pydantic import BaseModel

# Setup logging
logger = setup_logging("vokg-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler
    """
    # Startup
    logger.info("Starting VOKG API Gateway...")
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down VOKG API Gateway...")


# Create FastAPI app
app = FastAPI(
    title="VOKG API",
    description="Video Object Knowledge Graph - API Gateway",
    version="0.1.0",
    lifespan=lifespan,
)

# Setup middleware (CORS, logging, error handling)
setup_middleware(app)

# Include routers
app.include_router(health.router)
app.include_router(videos.router, prefix="/api/v1")
app.include_router(graphs.router, prefix="/api/v1")
app.include_router(queries.router, prefix="/api/v1")


# Authentication endpoints
class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@app.post("/api/v1/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user

    Args:
        request: Registration request
        db: Database session

    Returns:
        Access and refresh tokens
    """
    # Check if user exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        return JSONResponse(
            status_code=400,
            content={"detail": "Email already registered"},
        )

    existing_username = db.query(User).filter(User.username == request.username).first()
    if existing_username:
        return JSONResponse(
            status_code=400,
            content={"detail": "Username already taken"},
        )

    # Create user
    user = User(
        email=request.email,
        username=request.username,
        hashed_password=get_password_hash(request.password),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate tokens
    from backend.api_gateway.auth import create_refresh_token

    access_token = create_access_token({"sub": user.id})
    refresh_token = create_refresh_token({"sub": user.id})

    logger.info(f"User registered", user_id=user.id, email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login user

    Args:
        request: Login request
        db: Database session

    Returns:
        Access and refresh tokens
    """
    user = authenticate_user(db, request.email, request.password)

    if not user:
        return JSONResponse(
            status_code=401,
            content={"detail": "Incorrect email or password"},
        )

    # Generate tokens
    from backend.api_gateway.auth import create_refresh_token

    access_token = create_access_token({"sub": user.id})
    refresh_token = create_refresh_token({"sub": user.id})

    logger.info(f"User logged in", user_id=user.id, email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@app.get("/api/v1/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information

    Args:
        current_user: Authenticated user

    Returns:
        User information
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "is_active": current_user.is_active,
        "is_superuser": current_user.is_superuser,
        "created_at": current_user.created_at,
    }


# WebSocket endpoint for real-time updates
@app.websocket("/ws/{video_id}")
async def websocket_video_updates(
    websocket: WebSocket,
    video_id: int,
    token: str,
    db: Session = Depends(get_db),
):
    """
    WebSocket endpoint for video processing updates

    Args:
        websocket: WebSocket connection
        video_id: Video ID to subscribe to
        token: JWT authentication token
        db: Database session
    """
    try:
        # Authenticate user from token
        from backend.api_gateway.auth import get_user_from_token

        user = get_user_from_token(db, token)

        # Verify video ownership
        from backend.database.models import Video

        video = db.query(Video).filter(Video.id == video_id, Video.user_id == user.id).first()
        if not video:
            await websocket.close(code=4004, reason="Video not found")
            return

        # Handle WebSocket connection
        await websocket_endpoint(websocket, video_id, user.id)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=4000, reason="Authentication failed")


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "service": "VOKG API Gateway",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


# Run application
if __name__ == "__main__":
    uvicorn.run(
        "backend.api_gateway.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS if not settings.DEBUG else 1,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
