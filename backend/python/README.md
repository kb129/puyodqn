# PuyoDQN Backend - AI実装

深層Q学習（DQN）を使ったぷよぷよAIプレイヤーの実装です。

## 🎯 実装済み機能

### ✅ AIプレイヤー

1. **WeakCPU** - 無能AI
   - 最も深い谷を探して落とすだけの単純AI
   - デバッグ用・比較用として使用

2. **DQNPlayer** - 深層Q学習AI
   - 位置ベース行動空間（6列×4回転=24行動）
   - 動的マスキング技術で無効行動をフィルタリング
   - 444次元の拡張状態表現（盤面+戦術特徴+相対情報）

### ✅ 学習システム

- **自己対戦学習**：AIが自分と対戦して成長
- **密な報酬システム**：10要素の戦術的フィードバック
- **優先度付き経験再生**：効率的な学習データ活用
- **εグリーディ探索**：探索と活用のバランス

### ✅ ツール・API

- **FastAPI サーバー**：WebSocket + HTTP API
- **CLIツール**：AI管理・対戦・評価ツール
- **Docker対応**：uvベースのコンテナ化

## 🚀 クイックスタート

### 環境構築

```bash
# プロジェクトディレクトリに移動
cd backend/python

# UV環境でインストール（推奨）
uv sync

# PyTorchも含めて全インストール（初回のみ）
uv add torch torchvision

# 代替方法（uvがない場合）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 基本テスト

```bash
# AI コンポーネントテスト
uv run python cli.py test

# AI対戦テスト
uv run python cli.py battle weak weak --games 5

# テストスイート実行
uv run python test_ai.py
```

### サーバー起動

```bash
# 開発サーバー起動
uv run uvicorn server.main:app --reload --port 8000

# ヘルスチェック
curl http://localhost:8000/api/health
```

## 🤖 AI仕様

### 行動空間設計

- **行動数**: 24通り（6列 × 4回転）
- **行動ID**: `action_id = column * 4 + rotation`
- **動的マスキング**: 無効行動を自動除外

### 状態表現（444次元）

```
盤面状態:    390次元 (6×13×5色のone-hot)
現在ぷよ:     10次元 (2個×5色のone-hot)
ネクスト:     20次元 (4個×5色のone-hot)
基本情報:      4次元 (スコア、おじゃま、連鎖数、連鎖中)
戦術特徴:     12次元 (連鎖可能性、相性、脅威レベル等)
相対情報:      8次元 (対戦時の相手比較情報)
```

### 報酬設計

- **生存報酬**: 毎手0.01pt（基本生存インセンティブ）
- **配置品質**: 連結改善・バランス評価
- **連鎖報酬**: 連鎖数に応じたボーナス
- **勝敗報酬**: 勝利+10pt、敗北-10pt

## 📊 学習・評価

### 学習実行

```bash
# フル学習（10万エピソード）
uv run python ai/train_dqn.py --episodes 100000

# 短縮学習（テスト用）
uv run python ai/train_dqn.py --episodes 1000

# 評価のみ実行
uv run python ai/train_dqn.py --eval-only --eval-model models/best_model.pth
```

### 学習設定

```python
TrainingConfig:
  learning_rate: 1e-4
  batch_size: 32
  memory_size: 100,000
  epsilon_decay: 50,000 steps
  target_update: 2,000 steps
```

## 🔧 開発ツール

### CLIコマンド

```bash
# AI対戦
uv run python cli.py battle weak dqn --model-b models/best.pth --games 10

# 利用可能モデル一覧
uv run python cli.py list-models

# コンポーネントテスト
uv run python cli.py test

# 学習デモ
uv run python cli.py train-demo
```

### API エンドポイント

```bash
# CPU行動取得
POST /api/cpu/action
{
  "game_state": {...},
  "player_id": "A",
  "cpu_type": "weak|dqn",
  "model_path": "models/best.pth"
}

# WebSocket通信
WS /ws/game/{player_id}

# モデル管理
GET /api/ai/models
GET /api/ai/players
DELETE /api/ai/players/{key}
```

## 🐳 Docker使用

```bash
# コンテナビルド・起動
docker-compose up --build

# バックエンドのみ起動
docker-compose up --build backend
```

## 📁 ディレクトリ構造

```
backend/python/
├── ai/                    # AI実装
│   ├── base_player.py     # プレイヤー基底クラス
│   ├── weak_cpu.py        # 無能CPU
│   ├── dqn_player.py      # DQNプレイヤー
│   ├── dqn_trainer.py     # 学習システム
│   ├── game_adapter.py    # ゲーム統合
│   └── train_dqn.py       # 学習スクリプト
├── game_state/            # ゲーム状態管理
│   ├── board.py           # 盤面クラス
│   ├── puyo.py            # ぷよクラス
│   └── game_state.py      # 状態管理
├── server/                # API サーバー
│   └── main.py            # FastAPI アプリ
├── utils/                 # ユーティリティ
│   └── constants.py       # 定数定義
├── models/                # 学習済みモデル
├── logs/                  # ログファイル
├── cli.py                 # CLIツール
├── test_ai.py             # テストスイート
└── pyproject.toml         # プロジェクト設定
```

## 🎓 技術仕様

### ネットワーク構造

- **CNN層**: 盤面の空間パターン抽出
- **FC層**: 戦術特徴統合
- **出力**: 24行動のQ値

### 学習アルゴリズム

- **DQN**: Double DQN + Target Network
- **Experience Replay**: 優先度付きサンプリング
- **ε-greedy**: 動的探索率減衰

### 最適化

- **Adam Optimizer**: 学習率 1e-4
- **Gradient Clipping**: 勾配爆発防止
- **Batch Normalization**: 学習安定化

## ⚡ パフォーマンス

### 推論速度

- **CPU**: ~1-5ms/action
- **メモリ**: ~100MB（モデル込み）

### 学習速度

- **エピソード**: ~10-100 games/sec（速度調整可能）
- **収束**: ~10,000-50,000エピソード

## 🔮 将来拡張

### Phase 4+: 高度なAI

1. **マルチエージェント学習**
2. **階層強化学習**
3. **メタ学習**
4. **転移学習**

### 分析・可視化

1. **学習曲線可視化**
2. **戦術分析ダッシュボード**
3. **対戦リプレイシステム**
4. **AIの思考過程可視化**

## 📝 ライセンス

MIT License

---

🚀 **Deep Q-Learning meets Puyo Puyo!** 🎮