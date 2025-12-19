# 技術仕様書

## システムアーキテクチャ

### 全体構成
```
┌─────────────────┐    WebSocket/HTTP    ┌─────────────────┐
│   フロントエンド  │ ←─────────────────→ │  バックエンド    │
│  (React + TS)   │                     │   (Python)      │
│                 │                     │                 │
│ ┌─────────────┐ │                     │ ┌─────────────┐ │
│ │   ゲーム画面  │ │                     │ │  AI思考部    │ │
│ │  (Canvas)   │ │                     │ │  (CPU/DQN)  │ │
│ └─────────────┘ │                     │ └─────────────┘ │
│ ┌─────────────┐ │                     │ ┌─────────────┐ │
│ │ ゲームロジック │ │                     │ │ ゲーム状態   │ │
│ │  (TypeScript)│ │                     │ │ (共有モデル) │ │
│ └─────────────┘ │                     │ └─────────────┘ │
└─────────────────┘                     └─────────────────┘
```

### 技術スタック

#### フロントエンド
```yaml
言語: TypeScript
フレームワーク: React 18
ビルドツール: Vite
状態管理: Zustand
レンダリング: HTML5 Canvas 2D
スタイリング: CSS Modules / Styled Components
テスト: Jest + Testing Library
```

#### バックエンド  
```yaml
言語: Python 3.11+
パッケージマネージャー: uv (ultra-fast Python package manager)
フレームワーク: FastAPI
Webサーバー: Uvicorn
通信: WebSocket + HTTP
データ処理: NumPy
機械学習: PyTorch
テスト: pytest
リンター: Black + isort
```

#### 開発・デプロイツール
```yaml
フロントエンド:
  - Node.js + npm
  - Vite (開発サーバー・ビルド)
  - TypeScript コンパイラ
  
バックエンド:
  - uv (依存関係管理・仮想環境)
  - Docker + Docker Compose
  - uvicorn (ASGI サーバー)
  
共通:
  - Git (バージョン管理)
  - GitHub (リポジトリ・CI/CD)
```

### なぜuvを採用したか

uvは次世代Pythonパッケージマネージャーで、以下の利点があります：

#### 🚀 **速度**
- **10-100倍高速**: pip/poetryより圧倒的に高速な依存解決・インストール
- **並列処理**: 複数パッケージを並行してダウンロード・インストール
- **キャッシュ最適化**: 効率的なローカルキャッシュシステム

#### 🔒 **信頼性**
- **ロックファイル**: `uv.lock`で再現可能なビルド保証
- **依存解決**: より正確で高速な依存関係解決
- **隔離環境**: 完全に分離された仮想環境

#### 🛠 **開発効率**
- **シンプルコマンド**: `uv sync`, `uv run`, `uv add`で直感的操作
- **プロジェクト管理**: `pyproject.toml`ベースの統一設定
- **クロスプラットフォーム**: Windows/macOS/Linux対応

#### 📦 **現代的設計**
- **Rustで実装**: メモリ安全で高性能
- **PEP準拠**: Python標準に完全対応
- **pip互換**: 既存プロジェクトから簡単移行

```bash
# 従来のpipワークフロー (遅い)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # 30-60秒

# uvワークフロー (高速)
uv sync  # 3-5秒
```

## 🚀 開発環境セットアップ

### 前提条件

```bash
# 必須
- Node.js 18+ (フロントエンド)
- Python 3.11+ (バックエンド)
- Git

# 推奨
- uv (Pythonパッケージマネージャー)
- Docker Desktop (コンテナ実行用)
```

### 1. プロジェクトのクローン

```bash
git clone https://github.com/kb129/puyodqn.git
cd puyodqn
```

### 2. バックエンドセットアップ

```bash
cd backend/python

# uv使用（推奨・高速）
uv sync
uv run uvicorn server.main:app --reload --port 8000

# 従来方法（uvがない場合）
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn server.main:app --reload --port 8000
```

### 3. フロントエンドセットアップ

```bash
cd frontend

# 依存関係インストール
npm install

# 開発サーバー起動
npm run dev
```

### 4. 動作確認

```bash
# フロントエンド: http://localhost:5173/
# バックエンドAPI: http://localhost:8000/

# ヘルスチェック
curl http://localhost:8000/api/health
```

### 5. AI学習テスト

```bash
cd backend/python

# AIコンポーネントテスト
uv run python test_ai.py

# CLI操作テスト
uv run python cli.py test
uv run python cli.py battle weak weak --games 3
```

## データ構造

### 基本型定義

#### TypeScript側
```typescript
// 色定義
enum Color {
  EMPTY = 0,
  RED = 1,
  BLUE = 2,
  GREEN = 3,
  YELLOW = 4,
  OJAMA = 5
}

// ぷよペア
interface PuyoPair {
  colors: [Color, Color]; // [上, 下]
  x: number;              // x座標 (0-5)
  y: number;              // y座標 (0-12)
  rotation: number;       // 回転状態 (0-3)
}

// 盤面状態
interface BoardState {
  grid: Color[][];        // 6x13配列
  width: number;          // = 6
  height: number;         // = 13
}

// プレイヤー状態
interface PlayerState {
  id: 'A' | 'B';
  type: PlayerType;
  board: BoardState;
  fallingPuyo: PuyoPair | null;
  nextPuyos: PuyoPair[];  // [next, next2]
  score: number;
  isChaining: boolean;
  chainCount: number;
  ojamaPending: number;
}

// ゲーム状態
interface GameState {
  mode: 'single' | 'versus';
  players: PlayerState[];
  currentTurn: number;
  seed: number;           // 乱数シード
  gameOver: boolean;
  winner?: 'A' | 'B';
}
```

#### Python側
```python
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional

class Color(IntEnum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4
    OJAMA = 5

@dataclass
class PuyoPair:
    colors: Tuple[Color, Color]  # (上, 下)
    x: int                       # x座標 (0-5)
    y: int                       # y座標 (0-12)  
    rotation: int                # 回転状態 (0-3)

@dataclass
class BoardState:
    grid: List[List[Color]]      # 6x13配列
    width: int = 6
    height: int = 13

@dataclass  
class GameState:
    board: BoardState
    current_puyo: Optional[PuyoPair]
    next_puyos: List[PuyoPair]   # [next, next2]
    score: int
    is_chaining: bool
```

### 行動定義
```typescript
// フロントエンド→バックエンド
interface ActionRequest {
  type: 'get_action';
  game_state: GameState;
  player_id: 'A' | 'B';
  timestamp: number;
}

// バックエンド→フロントエンド  
interface ActionResponse {
  action: 'move_left' | 'move_right' | 'rotate_left' | 'rotate_right' | 'soft_drop' | 'no_action';
  confidence?: number;  // AI用（将来）
  thinking_time?: number;
}
```

## 通信プロトコル

### WebSocket通信
```typescript
// 接続エンドポイント
const wsUrl = 'ws://localhost:8000/game/cpu';

// メッセージ形式
interface WebSocketMessage {
  id: string;           // メッセージID
  type: string;         // メッセージタイプ  
  payload: any;         // データ本体
  timestamp: number;    // タイムスタンプ
}

// 例: CPUプレイヤーに行動要求
const message: WebSocketMessage = {
  id: 'req_001',
  type: 'get_cpu_action',
  payload: {
    game_state: currentGameState,
    player_id: 'B',
    difficulty: 'weak'
  },
  timestamp: Date.now()
};
```

### HTTP API（補助）
```python
# FastAPI エンドポイント
@app.post("/api/game/validate")
async def validate_game_state(state: GameState) -> dict:
    """ゲーム状態の検証"""
    pass

@app.post("/api/game/simulate")  
async def simulate_action(state: GameState, action: str) -> GameState:
    """行動シミュレーション"""
    pass
```

## ファイル構成

### フロントエンド
```
src/
├── components/                  # Reactコンポーネント
│   ├── Game/
│   │   ├── GameBoard.tsx       # ゲーム盤面
│   │   ├── NextDisplay.tsx     # ネクスト表示
│   │   ├── ScoreBoard.tsx      # スコア表示
│   │   └── OjamaPreview.tsx    # おじゃま予告
│   ├── Menu/
│   │   ├── MainMenu.tsx        # メイン画面
│   │   ├── PlayerSelect.tsx    # プレイヤー選択
│   │   └── GameOver.tsx        # ゲームオーバー
│   └── UI/
│       ├── Button.tsx          # 共通ボタン
│       └── DropDown.tsx        # ドロップダウン
├── game/                       # ゲームロジック
│   ├── Board.ts               # 盤面管理
│   ├── Puyo.ts                # ぷよクラス
│   ├── ChainCalculator.ts     # 連鎖計算
│   ├── ScoreCalculator.ts     # スコア計算
│   ├── GameEngine.ts          # ゲーム制御
│   └── PuyoGenerator.ts       # ぷよ生成
├── api/                       # 通信
│   ├── websocket.ts           # WebSocket管理
│   └── cpuPlayer.ts           # CPU通信
├── store/                     # 状態管理
│   ├── gameStore.ts           # ゲーム状態
│   └── settingsStore.ts       # 設定
├── utils/                     # ユーティリティ  
│   ├── constants.ts           # 定数定義
│   ├── animations.ts          # アニメーション
│   └── gameLogic.ts          # 共通ロジック
└── types/                     # 型定義
    ├── game.ts               # ゲーム型
    └── api.ts                # API型
```

### バックエンド
```python
python/
├── game_state/                # ゲーム状態表現
│   ├── __init__.py
│   ├── board.py              # 盤面クラス  
│   ├── puyo.py               # ぷよクラス
│   └── game_state.py         # 状態管理
├── ai/                       # AI実装
│   ├── __init__.py
│   ├── base_player.py        # プレイヤー基底
│   ├── weak_cpu.py           # 無能CPU
│   └── dqn_player.py         # DQN（将来）
├── server/                   # 通信サーバー
│   ├── __init__.py  
│   ├── main.py               # FastAPI アプリ
│   ├── websocket_handler.py  # WebSocket処理
│   └── game_api.py           # HTTP API
├── utils/                    # ユーティリティ
│   ├── __init__.py
│   ├── game_logic.py         # ゲームロジック
│   └── constants.py          # 定数
├── tests/                    # テスト
│   ├── test_board.py
│   ├── test_ai.py
│   └── test_api.py
└── requirements.txt          # 依存関係
```

## 開発・運用

### 開発環境
```yaml
# Docker Compose
services:
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    volumes: ["./frontend:/app"]
    
  backend:  
    build: ./python
    ports: ["8000:8000"]
    volumes: ["./python:/app"]
    environment:
      - PYTHONPATH=/app
```

### ビルド・デプロイ
```bash
# フロントエンド
cd frontend
npm run build
npm run preview

# バックエンド  
cd python
pip install -r requirements.txt
uvicorn server.main:app --reload

# テスト
npm test                    # フロントエンド
pytest                      # バックエンド
```

### パフォーマンス要件
- **レスポンス**: ユーザー入力への応答 < 16ms
- **AI思考時間**: CPU行動決定 < 100ms  
- **通信遅延**: WebSocket往復 < 50ms
- **メモリ使用量**: ブラウザ < 100MB、Python < 200MB
- **フレームレート**: 60fps維持

### 拡張性設計
- **AI追加**: `base_player.py`を継承して新AI実装
- **ルール変更**: 設定ファイルでパラメータ調整可能
- **マルチプレイ**: WebSocket経由でリモート対戦対応
- **学習データ**: ゲーム履歴をJSON/CSVで保存・分析