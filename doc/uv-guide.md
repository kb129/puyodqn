# 🚀 uv使用ガイド - 超高速Python開発

本プロジェクトでは、Pythonパッケージマネージャーとして **uv** を採用しています。

## 🌟 uvとは

uvは次世代Pythonパッケージマネージャーで、Rustで実装された超高速ツールです。

### 主な特徴

- ⚡ **10-100倍高速**: pip/poetryより圧倒的に高速
- 🔒 **再現可能**: `uv.lock`で完全な依存関係固定
- 🎯 **シンプル**: 直感的なコマンド体系
- 🛡 **安全**: Rustによるメモリ安全実装
- 📦 **互換性**: pip/requirements.txt完全対応

## 📥 uvインストール

### macOS / Linux

```bash
# Homebrewでインストール（推奨）
brew install uv

# curlでインストール  
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```bash
# PowerShell
irm https://astral.sh/uv/install.ps1 | iex

# Scoopでインストール
scoop install uv
```

### 確認

```bash
uv --version
# uv 0.5.4
```

## 🎯 基本コマンド

### プロジェクト操作

```bash
# 依存関係インストール・同期
uv sync                    # pyproject.toml + uv.lock から復元

# パッケージ追加
uv add fastapi             # 依存関係に追加
uv add pytest --dev       # 開発依存関係に追加

# パッケージ削除  
uv remove requests

# パッケージ一覧
uv list                    # インストール済み一覧
uv tree                    # 依存関係ツリー表示
```

### コマンド実行

```bash
# 仮想環境でコマンド実行
uv run python main.py
uv run pytest
uv run uvicorn server.main:app --reload

# 仮想環境アクティベート（必要時のみ）
source .venv/bin/activate  # Linux/macOS  
.venv\Scripts\activate     # Windows
```

### プロジェクト管理

```bash
# 新しいプロジェクト作成
uv init my-project
cd my-project

# 既存プロジェクトセットアップ
uv sync                    # 初回セットアップ
```

## 🔄 従来ツールとの比較

### 速度比較

| 操作 | pip | poetry | uv | 速度比 |
|------|-----|--------|----|----|
| 依存解決 | 60s | 30s | 2s | **30倍** |
| インストール | 45s | 25s | 3s | **15倍** |
| ロック生成 | - | 20s | 1s | **20倍** |

### コマンド対応表

| 操作 | pip | poetry | uv |
|------|-----|--------|----| 
| インストール | `pip install -r requirements.txt` | `poetry install` | `uv sync` |
| パッケージ追加 | `pip install requests` | `poetry add requests` | `uv add requests` |
| 実行 | `python main.py` | `poetry run python main.py` | `uv run python main.py` |
| 仮想環境 | `python -m venv venv` | `poetry shell` | 自動管理 |

## 📁 ファイル構成

uvプロジェクトは以下のファイルで管理されます：

```
project/
├── pyproject.toml         # プロジェクト設定・依存関係
├── uv.lock               # 完全な依存関係ロック（自動生成）
├── .venv/                # 仮想環境（自動生成）
└── README.md
```

### pyproject.toml例

```toml
[project]
name = "puyodqn-backend"
version = "1.0.0"
description = "PuyoDQN Backend AI System"
requires-python = ">=3.11"

dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0", 
    "websockets>=12.0",
    "numpy>=1.25.0",
    "torch>=2.0.0",
    "pydantic>=2.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "isort>=5.12",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "black>=23.0", 
    "isort>=5.12",
]
```

## 🛠 実用的なワークフロー

### 開発セットアップ

```bash
# 1. プロジェクトクローン
git clone https://github.com/kb129/puyodqn.git
cd puyodqn/backend/python

# 2. 依存関係インストール（3-5秒）
uv sync

# 3. 開発サーバー起動
uv run uvicorn server.main:app --reload
```

### パッケージ管理

```bash
# 新しい依存関係追加
uv add torch torchvision   # 本体依存関係
uv add pytest --dev       # 開発依存関係のみ

# 依存関係更新
uv sync --upgrade         # 全パッケージ更新
uv add "fastapi>=0.110"   # 特定パッケージ更新

# 依存関係確認
uv list                   # インストール済み一覧
uv tree                   # 依存関係ツリー
```

### テスト・品質管理

```bash
# テスト実行
uv run pytest
uv run python test_ai.py

# コード品質
uv run black .            # フォーマット
uv run isort .            # インポート整理
uv run mypy .             # 型チェック
```

### デプロイメント

```bash
# 本番用依存関係のみインストール
uv sync --no-dev

# requirements.txt生成（互換性用）
uv export --format requirements-txt > requirements.txt

# Docker用
uv export --format requirements-txt --no-dev > requirements-prod.txt
```

## 🔧 トラブルシューティング

### よくある問題

#### 1. uvコマンドが見つからない

```bash
# インストール確認
which uv
uv --version

# パス追加（必要時）
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### 2. 依存関係の競合

```bash
# ロックファイル再生成
rm uv.lock
uv sync

# 仮想環境リセット
rm -rf .venv
uv sync
```

#### 3. パッケージが見つからない

```bash
# キャッシュクリア
uv cache clean

# インデックス更新
uv sync --refresh
```

### デバッグオプション

```bash
# 詳細ログ出力
uv sync -v              # verbose
uv sync -vv             # 超詳細

# ドライラン（実行せず確認のみ）
uv add requests --dry-run
```

## 🔄 pip/poetryからの移行

### pipからの移行

```bash
# 既存のrequirements.txt活用
uv add --requirements requirements.txt

# pyproject.toml生成
uv init --name my-project
# requirements.txtの内容を手動でpyproject.tomlにコピー
```

### poetryからの移行

```bash
# pyproject.tomlはそのまま使える  
uv sync

# poetry.lockは不要（uv.lockが新しく生成される）
rm poetry.lock
```

## 🚀 パフォーマンス最適化

### 最速セットアップ

```bash
# 並列ダウンロード最大化
UV_CONCURRENT_DOWNLOADS=10 uv sync

# キャッシュ活用
uv sync --no-build-isolation  # ビルド分離無効化（高速）

# ネットワーク設定
UV_INDEX_URL=https://pypi.org/simple/ uv sync
```

### CI/CD最適化

```yaml
# GitHub Actions例
- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: |
    uv sync --no-dev
    uv run pytest
```

## 📚 参考リンク

- 📖 [公式ドキュメント](https://docs.astral.sh/uv/)
- 🏠 [GitHubリポジトリ](https://github.com/astral-sh/uv)  
- 📰 [リリースノート](https://github.com/astral-sh/uv/releases)
- 🛠 [設定リファレンス](https://docs.astral.sh/uv/configuration/)

---

⚡ **uvで爆速Python開発を体験しよう！** ⚡