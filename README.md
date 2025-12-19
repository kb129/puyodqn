# 🎮 PuyoDQN - AI学習対応ぷよぷよ

深層Q学習（DQN）を使ったぷよぷよAIプレイヤーの研究・開発プロジェクトです。

## 🎯 プロジェクト概要

本家ぷよぷよをコピーした実装に機械学習（DQN）による自動プレイヤーを組み込み、AIがぷよぷよの戦術を学習するシステムです。

### ✨ 主な特徴

- 🤖 **複数AIプレイヤー**: 無能CPU・DQN AI・人間プレイヤー
- 🏆 **自己対戦学習**: AIが自分と対戦して成長
- 📊 **学習管理システム**: リアルタイム進捗・結果分析
- 🎮 **完全なゲーム実装**: 連鎖・おじゃまぷよ・対戦モード
- ⚡ **高速開発環境**: uv + Vite で爆速セットアップ

## 🚀 クイックスタート

### 前提条件

```bash
# 必須ツール
Node.js 18+     # フロントエンド
Python 3.11+    # バックエンド
uv             # Python パッケージマネージャー（推奨）
```

### インストール・起動

```bash
# 1. プロジェクトクローン
git clone https://github.com/kb129/puyodqn.git
cd puyodqn

# 2. バックエンド起動
cd backend/python
uv sync                                             # 依存関係インストール (3-5秒)
uv run uvicorn server.main:app --reload --port 8000

# 3. フロントエンド起動（新しいターミナル）
cd ../../frontend
npm install && npm run dev                          # http://localhost:5173/
```

### 動作確認

1. **ゲーム**: http://localhost:5173/ でぷよぷよをプレイ
2. **API**: http://localhost:8000/api/health でバックエンド確認
3. **AI対戦**: プレイヤー選択で「DQN AI」を選択

## 🏗 技術スタック

### フロントエンド
- **React 18** + **TypeScript** - モダンWeb開発
- **Vite** - 高速ビルドツール
- **Zustand** - 軽量状態管理
- **Canvas 2D** - ゲームレンダリング

### バックエンド
- **Python 3.11+** + **FastAPI** - 高性能APIサーバー
- **uv** - 超高速パッケージマネージャー (pip の10-100倍高速)
- **PyTorch** - 深層学習フレームワーク
- **WebSocket** - リアルタイム通信

### AI・機械学習
- **DQN (Deep Q-Network)** - 深層強化学習
- **自己対戦学習** - AI vs AI で効率学習
- **位置ベース行動空間** - 24行動（6列×4回転）
- **密な報酬システム** - 10要素の戦術的フィードバック

## 🎮 ゲーム機能

### ✅ 実装済み
- ✅ 基本ゲームシステム（落下・回転・移動）
- ✅ 連鎖システム（4個以上同色で消去）
- ✅ おじゃまぷよシステム
- ✅ 対戦モード（人間 vs 人間/AI）
- ✅ 無能CPU（シンプルAI）
- ✅ DQN AI（深層学習AI）

### 🚧 開発中
- 🚧 AI学習システム（フロントエンド完成・バックエンド実装中）
- 🚧 学習進捗可視化
- 🚧 AIトーナメントシステム

## 🤖 AI仕様

### DQNプレイヤー
```python
# 行動空間: 24通り (6列 × 4回転)
# 状態空間: 444次元
#   - 盤面状態: 390次元 (6×13×5色)
#   - ぷよ情報: 30次元 (現在+ネクスト)  
#   - 戦術特徴: 24次元 (連鎖・相性・脅威)

# 報酬設計
rewards = {
    "survival": 0.01,      # 毎手生存
    "chain": chain_bonus,  # 連鎖実行
    "position": quality,   # 配置品質
    "win": 10.0,          # 勝利
    "lose": -10.0         # 敗北
}
```

### 学習システム
- **自己対戦**: 同一モデルのコピー同士が対戦
- **経験再生**: 優先度付きリプレイバッファ
- **動的マスキング**: 無効行動を自動除外
- **カリキュラム学習**: 段階的難易度調整

## 📊 学習・評価

### AI学習実行

```bash
cd backend/python

# 短縮学習（テスト用）
uv run python ai/train_dqn.py --episodes 1000

# フル学習（本格的）
uv run python ai/train_dqn.py --episodes 100000

# AI対戦テスト
uv run python cli.py battle weak dqn --games 10
```

### Webベース学習管理

1. フロントエンドで「AI学習モード」選択
2. 学習設定（エピソード数・ハイパーパラメータ）
3. リアルタイム進捗監視
4. 結果分析・モデル保存

## 📁 プロジェクト構造

```
puyodqn/
├── frontend/               # React + TypeScript
│   ├── src/components/     # ゲーム画面・学習管理画面
│   ├── src/store/         # Zustand状態管理
│   └── src/types/         # 型定義
├── backend/python/         # FastAPI + PyTorch  
│   ├── ai/                # AI実装（WeakCPU・DQN）
│   ├── server/            # APIサーバー
│   ├── game_state/        # ゲームロジック
│   └── models/            # 学習済みモデル
├── doc/                   # 仕様書・ドキュメント
└── docker-compose.yml     # コンテナ構成
```

## 🛠 開発コマンド

### バックエンド開発

```bash
cd backend/python

# 開発環境
uv sync                    # 依存関係更新
uv run uvicorn server.main:app --reload

# テスト・検証
uv run python test_ai.py   # AIテストスイート
uv run python cli.py test  # コンポーネントテスト
uv run black .             # コードフォーマット

# AI管理
uv run python cli.py list-models     # モデル一覧
uv run python cli.py battle weak dqn # AI対戦
```

### フロントエンド開発

```bash
cd frontend

# 開発環境
npm run dev               # 開発サーバー
npm run build             # プロダクションビルド
npm run preview           # ビルド結果プレビュー

# 品質管理
npm run type-check        # TypeScriptチェック
npm run lint              # ESLint
```

## 🐳 Docker使用

```bash
# 全サービス起動
docker-compose up --build

# バックエンドのみ
docker-compose up --build backend

# クリーンアップ
docker-compose down --volumes
```

## 📖 ドキュメント

詳細な仕様は `doc/` ディレクトリをご参照ください：

- [📋 全体仕様](./doc/README.md)
- [🎮 ゲームルール](./doc/game-rules.md)  
- [🖥 画面仕様](./doc/screen-spec.md)
- [⚙️ 技術仕様](./doc/technical-spec.md)
- [🤖 AI仕様](./doc/ai-spec.md)
- [🔄 API仕様](./doc/api-spec.md)

## 🎯 開発フェーズ

### ✅ Phase 1: 基盤構築 (完了)
- React + Canvas ゲーム実装
- Python通信サーバー
- データ交換システム

### ✅ Phase 2: 無能CPU (完了)  
- 基本AI実装
- フロントエンド・バックエンド連携

### ✅ Phase 3: ゲーム完成 (完了)
- 連鎖・スコア計算
- おじゃまぷよシステム
- 対戦機能

### 🚧 Phase 4: 機械学習 (進行中)
- ✅ DQNプレイヤー実装
- ✅ 学習システム設計
- 🚧 学習管理UI
- 🚧 性能評価システム

### 🔮 Phase 5: 高度なAI (予定)
- マルチエージェント学習
- 階層強化学習  
- メタ学習・転移学習
- 思考過程可視化

## 🤝 コントリビューション

プルリクエスト・Issues歓迎！特に以下の分野：

- 🎨 UI/UX改善
- 🧠 AI アルゴリズム最適化
- 📊 学習データ可視化
- 🎮 新ゲームモード追加
- 📝 ドキュメント改善

## 📄 ライセンス

MIT License

---

🚀 **Deep Q-Learning meets Puyo Puyo!** 🎮

**強化学習でぷよぷよマスターを目指そう！**