import logging
import json
import asyncio
from typing import Dict, List, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.core.security import decode_access_token
from app.database.session import SessionLocal
from app.models.user import User
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["websockets"])

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)

    async def send_personal_message(self, message: Dict[str, Any], user_id: int):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time UI updates.
    """
    db = SessionLocal()
    try:
        # Authenticate user via token
        token_data = decode_access_token(token)
        if not token_data:
            await websocket.close(code=1008)
            return
            
        user = db.query(User).filter(User.email == token_data.email).first()
        if not user:
            await websocket.close(code=1008)
            return
            
        user_id = user.id
    except Exception:
        await websocket.close(code=1008)
        return
    finally:
        db.close()

    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive data from client (e.g., requests for streaming advice)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "stream_advice":
                # Implementation of token-level streaming
                await stream_ai_advice(websocket, message.get("query"), user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)

async def stream_ai_advice(websocket: WebSocket, query: str, user_id: int):
    """Stream LLM advice token-by-token for perceived speed."""
    llm_service = LLMService()
    
    try:
        async for chunk in llm_service.stream_chat_async(
            messages=[{"role": "user", "content": query}]
        ):
            await websocket.send_json({
                "type": "advice_chunk",
                "content": chunk,
                "status": "processing"
            })
        
        await websocket.send_json({
            "type": "advice_chunk",
            "content": "",
            "status": "completed"
        })
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})

# Helper to notify UI from background workers
async def notify_user_status(user_id: int, status: str, data: Any = None):
    await manager.send_personal_message({
        "type": "status_update",
        "status": status,
        "data": data
    }, user_id)
