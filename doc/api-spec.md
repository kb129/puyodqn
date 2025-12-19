# API仕様書

## 通信アーキテクチャ

### 概要
フロントエンド（React/TypeScript）とバックエンド（Python/FastAPI）間の通信は、主にWebSocketを使用してリアルタイム性を確保します。

```
Frontend (React/TS) ←──WebSocket──→ Backend (Python/FastAPI)
                    ←───HTTP────→
```

## WebSocket API

### 接続エンドポイント
```
ws://localhost:8000/ws/game/{player_id}
```

### メッセージ基本形式
```json
{
  "id": "string",           // ユニークなメッセージID
  "type": "string",         // メッセージタイプ
  "payload": {},            // データ本体
  "timestamp": 1640995200   // Unix timestamp
}
```

### メッセージタイプ

#### 1. CPU行動要求 (`get_cpu_action`)
**送信**: Frontend → Backend

```json
{
  "id": "req_001",
  "type": "get_cpu_action", 
  "payload": {
    "game_state": {
      "board": [[0,0,0,0,0,0], ...],  // 6x13配列
      "current_puyo": {
        "colors": [1, 2],              // [上, 下]の色
        "x": 2,                        // 現在x座標
        "y": 0,                        // 現在y座標  
        "rotation": 0                  // 回転状態(0-3)
      },
      "next_puyos": [[1,2], [3,4]],    // [next, next2]
      "score": 0,
      "is_chaining": false
    },
    "player_id": "B",
    "cpu_type": "weak"
  },
  "timestamp": 1640995200
}
```

**応答**: Backend → Frontend
```json
{
  "id": "req_001",
  "type": "cpu_action_response",
  "payload": {
    "action": "move_left",             // 行動種別
    "confidence": 0.8,                 // 信頼度（将来のAI用）
    "thinking_time": 50,               // 思考時間(ms)
    "debug_info": {                    // デバッグ情報
      "evaluated_positions": 6,
      "best_column": 2,
      "depth_scores": [3, 5, 8, 4, 2, 6]
    }
  },
  "timestamp": 1640995250
}
```

#### 2. ゲーム状態同期 (`sync_game_state`)
**送信**: Frontend → Backend

```json
{
  "id": "sync_001",
  "type": "sync_game_state",
  "payload": {
    "game_mode": "versus",
    "players": [
      {
        "id": "A",
        "type": "human",
        "board": [[0,0,0,0,0,0], ...],
        "score": 1200,
        "is_chaining": false,
        "ojama_pending": 3
      },
      {
        "id": "B", 
        "type": "cpu_weak",
        "board": [[0,0,0,0,0,0], ...],
        "score": 800,
        "is_chaining": true,
        "ojama_pending": 0
      }
    ],
    "seed": 12345678,
    "turn": 42
  },
  "timestamp": 1640995200
}
```

#### 3. エラー応答 (`error`)
```json
{
  "id": "req_001",
  "type": "error",
  "payload": {
    "code": "INVALID_GAME_STATE",
    "message": "Board dimensions invalid: expected 6x13",
    "details": {
      "received_width": 5,
      "received_height": 12
    }
  },
  "timestamp": 1640995200
}
```

### 行動タイプ定義
```typescript
type Action = 
  | 'move_left'      // 左移動
  | 'move_right'     // 右移動  
  | 'rotate_left'    // 左回転
  | 'rotate_right'   // 右回転
  | 'soft_drop'      // ソフトドロップ
  | 'no_action';     // 行動なし
```

## HTTP API

### ベースURL
```
http://localhost:8000/api
```

### エンドポイント一覧

#### 1. ゲーム状態検証
```http
POST /api/game/validate
Content-Type: application/json
```

**リクエスト**:
```json
{
  "board": [[0,0,0,0,0,0], ...],
  "current_puyo": {"colors": [1,2], "x": 2, "y": 0, "rotation": 0},
  "next_puyos": [[1,2], [3,4]]
}
```

**レスポンス**:
```json
{
  "valid": true,
  "errors": [],
  "warnings": ["Board nearly full at column 2"]
}
```

#### 2. 行動シミュレーション  
```http
POST /api/game/simulate
Content-Type: application/json
```

**リクエスト**:
```json
{
  "game_state": { /* GameState */ },
  "action": "move_left",
  "steps": 1
}
```

**レスポンス**:  
```json
{
  "result_state": { /* GameState */ },
  "chain_occurred": false,
  "score_gained": 0,
  "ojama_sent": 0
}
```

#### 3. CPU設定取得
```http
GET /api/cpu/config/{cpu_type}
```

**レスポンス**:
```json
{
  "type": "weak", 
  "description": "Simple depth-based CPU",
  "parameters": {
    "reaction_time_ms": 100,
    "look_ahead_depth": 1,
    "randomness": 0.1
  }
}
```

#### 4. ヘルスチェック
```http
GET /api/health
```

**レスポンス**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "active_connections": 2
}
```

## データ型定義

### TypeScript型
```typescript
// 共通インターフェース
interface ApiMessage<T = any> {
  id: string;
  type: string;  
  payload: T;
  timestamp: number;
}

// WebSocket専用型
interface WebSocketClient {
  send(message: ApiMessage): void;
  onMessage(callback: (message: ApiMessage) => void): void;
  onError(callback: (error: Error) => void): void;
  close(): void;
}

// HTTP専用型  
interface HttpClient {
  post<T, R>(url: string, data: T): Promise<R>;
  get<R>(url: string): Promise<R>;
}
```

### Python型
```python
from pydantic import BaseModel
from typing import List, Optional, Union
from enum import Enum

class ActionType(str, Enum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right" 
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    SOFT_DROP = "soft_drop"
    NO_ACTION = "no_action"

class GameStateRequest(BaseModel):
    board: List[List[int]]
    current_puyo: Optional[dict]
    next_puyos: List[List[int]]
    score: int
    is_chaining: bool

class ActionRequest(BaseModel):
    game_state: GameStateRequest  
    player_id: str
    cpu_type: str

class ActionResponse(BaseModel):
    action: ActionType
    confidence: Optional[float] = None
    thinking_time: Optional[int] = None
    debug_info: Optional[dict] = None
```

## 実装例

### フロントエンド（WebSocket クライアント）
```typescript
class GameWebSocketClient {
  private ws: WebSocket;
  private messageCallbacks = new Map<string, (response: ApiMessage) => void>();

  constructor(playerId: string) {
    this.ws = new WebSocket(`ws://localhost:8000/ws/game/${playerId}`);
    this.setupEventHandlers();
  }

  async requestCpuAction(gameState: GameState): Promise<ActionResponse> {
    const messageId = `req_${Date.now()}`;
    
    const request: ApiMessage = {
      id: messageId,
      type: 'get_cpu_action',
      payload: { game_state: gameState, player_id: 'B', cpu_type: 'weak' },
      timestamp: Date.now()
    };

    return new Promise((resolve, reject) => {
      this.messageCallbacks.set(messageId, (response) => {
        if (response.type === 'error') {
          reject(new Error(response.payload.message));
        } else {
          resolve(response.payload);
        }
      });

      this.ws.send(JSON.stringify(request));
      
      // タイムアウト設定
      setTimeout(() => {
        if (this.messageCallbacks.has(messageId)) {
          this.messageCallbacks.delete(messageId);
          reject(new Error('Request timeout'));
        }
      }, 5000);
    });
  }

  private setupEventHandlers() {
    this.ws.onmessage = (event) => {
      const message: ApiMessage = JSON.parse(event.data);
      const callback = this.messageCallbacks.get(message.id);
      
      if (callback) {
        callback(message);
        this.messageCallbacks.delete(message.id);
      }
    };
  }
}
```

### バックエンド（FastAPI WebSocket ハンドラ）
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import asyncio

app = FastAPI()

@app.websocket("/ws/game/{player_id}")
async def websocket_endpoint(websocket: WebSocket, player_id: str):
    await websocket.accept()
    
    try:
        while True:
            # メッセージ受信
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # メッセージタイプに応じて処理
            if message['type'] == 'get_cpu_action':
                response = await handle_cpu_action_request(message)
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        print(f"Player {player_id} disconnected")

async def handle_cpu_action_request(message: dict) -> dict:
    # CPU思考処理
    game_state = message['payload']['game_state']
    cpu_type = message['payload']['cpu_type']
    
    # 無能CPU実装
    if cpu_type == 'weak':
        from ai.weak_cpu import WeakCPU
        cpu = WeakCPU()
        action = cpu.get_action(game_state)
    
    return {
        'id': message['id'],
        'type': 'cpu_action_response',
        'payload': {
            'action': action,
            'thinking_time': 50
        },
        'timestamp': int(time.time())
    }
```

## エラーハンドリング

### エラーコード一覧
```typescript
enum ApiErrorCode {
  INVALID_GAME_STATE = 'INVALID_GAME_STATE',
  INVALID_ACTION = 'INVALID_ACTION',  
  CPU_TIMEOUT = 'CPU_TIMEOUT',
  CONNECTION_ERROR = 'CONNECTION_ERROR',
  INTERNAL_ERROR = 'INTERNAL_ERROR'
}
```

### リトライ戦略
```typescript
class ApiClient {
  private async withRetry<T>(
    operation: () => Promise<T>, 
    maxRetries: number = 3
  ): Promise<T> {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await operation();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        await this.delay(Math.pow(2, i) * 1000); // Exponential backoff
      }
    }
    throw new Error('Max retries exceeded');
  }
}
```

## パフォーマンス仕様

### レスポンス時間要件
- **CPU行動要求**: < 100ms
- **状態同期**: < 50ms  
- **HTTP API**: < 200ms
- **WebSocket接続**: < 1s

### スループット要件
- **同時接続**: 100クライアント
- **メッセージ/秒**: 1000件
- **帯域幅**: 10Mbps

### モニタリング
```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000
        
        # ログ出力
        print(f"{func.__name__}: {duration:.2f}ms")
        
        return result
    return wrapper
```