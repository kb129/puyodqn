"""FastAPI メインサーバー"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import time
import asyncio

from game_state.game_state import GameState
from ai.weak_cpu import WeakCPU
from ai.dqn_player import DQNPlayer
from ai.base_player import BasePlayer

app = FastAPI(title="PuyoDQN Backend API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Viteのデフォルトポート
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル状態
active_connections: Dict[str, WebSocket] = {}
ai_players: Dict[str, BasePlayer] = {}

# DQNモデルのパス設定
DEFAULT_DQN_MODEL_PATH = "models/best_model.pth"

def create_ai_player(player_id: str, cpu_type: str, model_path: Optional[str] = None) -> BasePlayer:
    """AIプレイヤーを作成"""
    if cpu_type == "weak":
        return WeakCPU(player_id)
    elif cpu_type == "dqn":
        model_file = model_path or DEFAULT_DQN_MODEL_PATH
        try:
            # ファイルが存在しない場合はWeakCPUにフォールバック
            import os
            if not os.path.exists(model_file):
                print(f"Warning: DQN model not found at {model_file}, using WeakCPU instead")
                return WeakCPU(player_id)
            
            dqn_player = DQNPlayer(player_id, model_path=model_file)
            dqn_player.epsilon = 0.0  # 評価モード（探索なし）
            return dqn_player
        except Exception as e:
            print(f"Error loading DQN player: {e}, using WeakCPU instead")
            return WeakCPU(player_id)
    else:
        raise ValueError(f"Unknown CPU type: {cpu_type}")

class GameStateRequest(BaseModel):
    """ゲーム状態リクエスト"""
    board: list
    current_puyo: Optional[dict]
    next_puyos: list
    score: int
    is_chaining: bool

class ActionRequest(BaseModel):
    """行動リクエスト"""
    game_state: dict
    player_id: str
    cpu_type: str = "weak"  # "weak", "dqn"
    model_path: Optional[str] = None

class ApiMessage(BaseModel):
    """API メッセージ基本形式"""
    id: str
    type: str
    payload: dict
    timestamp: int

@app.get("/")
async def root():
    """ヘルスチェック"""
    return {"status": "healthy", "message": "PuyoDQN Backend is running"}

@app.get("/api/health")
async def health_check():
    """詳細ヘルスチェック"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_connections": len(active_connections),
        "active_ai_players": len(ai_players),
        "uptime": time.time()
    }

@app.get("/api/ai/players")
async def list_ai_players():
    """アクティブなAIプレイヤー一覧"""
    players_info = []
    for key, player in ai_players.items():
        players_info.append({
            "key": key,
            "player_id": player.player_id,
            "name": player.name,
            "type": "DQN" if isinstance(player, DQNPlayer) else "CPU"
        })
    
    return {"players": players_info}

@app.delete("/api/ai/players/{player_key}")
async def remove_ai_player(player_key: str):
    """AIプレイヤーを削除"""
    if player_key in ai_players:
        del ai_players[player_key]
        return {"message": f"Player {player_key} removed"}
    else:
        raise HTTPException(status_code=404, detail="Player not found")

@app.get("/api/ai/models")
async def list_available_models():
    """利用可能なモデル一覧"""
    import os
    import glob
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {"models": []}
    
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    models_info = []
    
    for model_file in model_files:
        file_stat = os.stat(model_file)
        models_info.append({
            "filename": os.path.basename(model_file),
            "path": model_file,
            "size": file_stat.st_size,
            "modified": file_stat.st_mtime
        })
    
    # 更新日時でソート
    models_info.sort(key=lambda x: x["modified"], reverse=True)
    
    return {"models": models_info}

@app.post("/api/cpu/action")
async def get_cpu_action(request: ActionRequest):
    """CPU行動取得（HTTP版）"""
    try:
        # CPU プレイヤー取得または作成
        cpu_key = f"{request.player_id}_{request.cpu_type}"
        if cpu_key not in ai_players:
            ai_players[cpu_key] = create_ai_player(request.player_id, request.cpu_type, request.model_path)
        
        cpu = ai_players[cpu_key]
        
        # 行動決定
        start_time = time.time()
        action = cpu.get_action(request.game_state)
        thinking_time = int((time.time() - start_time) * 1000)
        
        # デバッグ情報（WeakCPUのみ）
        debug_info = {}
        if hasattr(cpu, 'get_debug_info'):
            debug_info = cpu.get_debug_info(request.game_state)
        
        return {
            "action": action,
            "thinking_time": thinking_time,
            "debug_info": debug_info,
            "cpu_type": request.cpu_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CPU action failed: {str(e)}")

@app.websocket("/ws/game/{player_id}")
async def websocket_endpoint(websocket: WebSocket, player_id: str):
    """WebSocket ゲーム通信エンドポイント"""
    await websocket.accept()
    active_connections[player_id] = websocket
    
    try:
        while True:
            # メッセージ受信
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # メッセージ処理
            response = await handle_websocket_message(message, player_id)
            
            # レスポンス送信
            if response:
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        print(f"Player {player_id} disconnected")
    except Exception as e:
        print(f"WebSocket error for player {player_id}: {e}")
        error_response = {
            "id": message.get("id", "unknown"),
            "type": "error",
            "payload": {
                "code": "WEBSOCKET_ERROR",
                "message": str(e)
            },
            "timestamp": int(time.time())
        }
        try:
            await websocket.send_text(json.dumps(error_response))
        except:
            pass
    finally:
        if player_id in active_connections:
            del active_connections[player_id]

async def handle_websocket_message(message: dict, player_id: str) -> Optional[dict]:
    """WebSocket メッセージハンドラ"""
    message_type = message.get("type")
    message_id = message.get("id")
    payload = message.get("payload", {})
    
    try:
        if message_type == "get_cpu_action":
            return await handle_cpu_action_request(message_id, payload, player_id)
        elif message_type == "ping":
            return {
                "id": message_id,
                "type": "pong",
                "payload": {"timestamp": int(time.time())},
                "timestamp": int(time.time())
            }
        else:
            return {
                "id": message_id,
                "type": "error",
                "payload": {
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {message_type}"
                },
                "timestamp": int(time.time())
            }
            
    except Exception as e:
        return {
            "id": message_id,
            "type": "error", 
            "payload": {
                "code": "MESSAGE_HANDLER_ERROR",
                "message": str(e)
            },
            "timestamp": int(time.time())
        }

async def handle_cpu_action_request(message_id: str, payload: dict, player_id: str) -> dict:
    """CPU行動リクエスト処理"""
    game_state = payload.get("game_state")
    cpu_type = payload.get("cpu_type", "weak")
    target_player = payload.get("player_id", player_id)
    model_path = payload.get("model_path")
    
    if not game_state:
        raise ValueError("game_state is required")
    
    # CPU プレイヤー取得または作成
    cpu_key = f"{target_player}_{cpu_type}"
    if cpu_key not in ai_players:
        ai_players[cpu_key] = create_ai_player(target_player, cpu_type, model_path)
    
    cpu = ai_players[cpu_key]
    
    # 行動決定（少し遅延を入れて人間らしくする）
    start_time = time.time()
    if cpu_type == "weak":
        await asyncio.sleep(0.05)  # 50ms の遅延（WeakCPU用）
    
    action = cpu.get_action(game_state)
    thinking_time = int((time.time() - start_time) * 1000)
    
    # デバッグ情報（WeakCPUのみ）
    debug_info = {}
    if hasattr(cpu, 'get_debug_info'):
        debug_info = cpu.get_debug_info(game_state)
    
    return {
        "id": message_id,
        "type": "cpu_action_response",
        "payload": {
            "action": action,
            "thinking_time": thinking_time,
            "debug_info": debug_info,
            "cpu_type": cpu_type
        },
        "timestamp": int(time.time())
    }

@app.post("/api/game/validate")
async def validate_game_state(game_state: GameStateRequest):
    """ゲーム状態検証"""
    errors = []
    warnings = []
    
    # 盤面サイズチェック
    if len(game_state.board) != 13:
        errors.append(f"Invalid board height: {len(game_state.board)}, expected 13")
    
    for i, row in enumerate(game_state.board):
        if len(row) != 6:
            errors.append(f"Invalid row {i} width: {len(row)}, expected 6")
    
    # 警告チェック（盤面が満杯に近い）
    for col in range(6):
        height = 0
        for row in range(12, -1, -1):
            if game_state.board[row][col] != 0:
                height = 13 - row
                break
        
        if height > 10:
            warnings.append(f"Column {col} is nearly full (height: {height})")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)