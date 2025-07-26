# RefImage - AI-Powered Image Store and Search Engine

RefImageは、OpenAI CLIPモデルを使用したAI搭載の画像ストレージおよび検索エンジンです。自然言語クエリによる高精度な画像検索機能を提供します。

## 主な機能

### 🔍 AI画像検索
- **自然言語検索**: "red car", "sunset landscape"などの自然な文章で画像を検索
- **セマンティック検索**: 画像の内容を理解した意味的な検索
- **高精度マッチング**: CLIP embeddings による高精度な類似度計算

### 🗃️ 画像管理
- **メタデータ管理**: 説明、タグ、撮影日時などの詳細情報
- **タグシステム**: 柔軟なタグベースの分類とフィルタリング
- **ファイル管理**: 安全な画像ファイルストレージとアクセス

### 🔧 高度なクエリ機能
- **DSL (Dynamic Search Language)**: 複雑な検索条件の指定
- **論理演算子**: AND, OR, NOTを使った組み合わせ検索
- **複合クエリ**: テキスト検索とタグフィルタの組み合わせ

### 🚀 REST API
- **RESTful API**: 標準的なHTTP APIによる操作
- **OpenAPI対応**: Swagger UIによるインタラクティブなドキュメント
- **型安全**: Pydantic v2による厳密な型定義

## 技術スタック

### バックエンド
- **Python 3.10+**: メインプログラミング言語
- **FastAPI**: 高性能Webフレームワーク
- **Pydantic v2**: データバリデーションとシリアライゼーション

### AI/ML
- **OpenAI CLIP**: 画像・テキスト埋め込みモデル
- **PyTorch**: 機械学習フレームワーク
- **FAISS**: 高速ベクトル類似度検索

### データベース・ストレージ
- **SQLite**: メタデータストレージ
- **ファイルシステム**: 画像ファイルストレージ
- **JSON**: 埋め込みベクトルシリアライゼーション

- **Image Upload & Storage**: Upload images and automatically generate CLIP embeddings
- **Semantic Search**: Search for images using natural language text queries
- **Vector Indexing**: Fast similarity search using FAISS
- **RESTful API**: Built with FastAPI for high performance
- **Advanced Query DSL**: Support for complex search patterns (planned)
- **LLM Integration**: Dynamic query composition via external LLMs (planned)

## Tech Stack

- **Python 3.9+**: Core language
- **FastAPI**: Web framework for API endpoints
- **PyTorch**: Deep learning framework for CLIP model
- **CLIP**: OpenAI's multimodal model for image-text embeddings
- **FAISS**: Facebook's similarity search library
- **Pydantic**: Data validation and serialization

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd refimage

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For GPU support (optional)
pip install -e ".[gpu]"
```

### Running the Server

```bash
# Start the development server
uvicorn refimage.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive documentation can be accessed at `http://localhost:8000/docs`.

## API Endpoints

### Upload Image
```http
POST /images
Content-Type: multipart/form-data

{
  "file": <image_file>
}
```

### Search Images
```http
GET /search?q=<text_query>&limit=10
```

### Advanced Search (planned)
```http
POST /search
Content-Type: application/json

{
  "query": "<DSL_query>",
  "limit": 10
}
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
isort src tests

# Type checking
mypy src

# Linting
flake8 src tests
```

### Project Structure

```
src/refimage/
├── __init__.py
├── main.py              # FastAPI application entry point
├── api/                 # API endpoints
│   ├── __init__.py
│   ├── images.py        # Image upload endpoints
│   └── search.py        # Search endpoints
├── core/                # Core business logic
│   ├── __init__.py
│   ├── embedding.py     # CLIP embedding generation
│   ├── indexing.py      # FAISS vector indexing
│   └── search.py        # Search logic
├── models/              # Pydantic data models
│   ├── __init__.py
│   ├── image.py         # Image-related models
│   └── search.py        # Search-related models
├── storage/             # Data storage
│   ├── __init__.py
│   ├── image_store.py   # Image file storage
│   └── metadata.py     # Metadata storage
└── config.py            # Configuration settings
```

## Configuration

The application uses environment variables for configuration:

```env
# Model settings
CLIP_MODEL_NAME=ViT-B/32
DEVICE=auto  # auto, cpu, cuda

# Storage settings
IMAGE_STORAGE_PATH=./data/images
INDEX_STORAGE_PATH=./data/indexes
METADATA_STORAGE_PATH=./data/metadata

# API settings
MAX_IMAGE_SIZE=10485760  # 10MB
ALLOWED_IMAGE_TYPES=jpg,jpeg,png,webp

# FAISS settings
INDEX_TYPE=flat  # flat, ivf, hnsw
SEARCH_K=100
```

## 📊 Code Quality & Development

RefImageプロジェクトでは、高品質なコードベースを維持するために包括的な自動化ツールを導入しています。

### 🔧 Quality Check Tools

```bash
# 全品質チェック実行
make quality-check

# 自動修正付き品質チェック
make quality-fix

# 個別ツール実行
make format        # Black + isort (コードフォーマット)
make lint          # Flake8 (構文・スタイルチェック)
make type-check    # MyPy (型チェック)
make security      # Bandit (セキュリティスキャン)
make duplication   # jscpd (重複コードチェック)
```

### 📈 Quality Metrics

- **Code Duplication**: < 1% (目標)
- **Type Coverage**: MyPy strict mode
- **Security Scan**: Bandit medium+ severity
- **Code Style**: Black + isort + Flake8
- **Pre-commit Hooks**: 自動品質チェック

### 🚀 CI/CD Integration

GitHub Actionsによる自動品質チェック:
- Python 3.9, 3.10, 3.11でのマルチバージョンテスト
- Pull Request時の自動品質レポート
- セキュリティ脆弱性の継続監視
- コード品質メトリクスの追跡

### 🛠️ Development Setup

```bash
# 開発環境セットアップ
pip install -e ".[dev]"
npm install -g jscpd
pre-commit install

# 品質チェック自動化
pre-commit run --all-files
```

## License

MIT License - see [LICENSE](LICENSE) for details.
