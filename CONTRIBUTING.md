# 🦛 Contributing to Chonkie

> **PRs are frozen.** We're building Chonkie v2. v1.7 is the last 1.x release; fixes go into 2.0. New pull requests on `main` will be closed automatically until v2 is stable. Please bear with us.

> "I like them big, I like them CONTRIBUTING" ~ Moto Moto, probably

Welcome fellow CHONKer! We're thrilled you want to contribute to Chonkie. Every contribution—whether fixing bugs, adding features, or improving documentation—makes Chonkie better for everyone.

## 🚀 Getting Started

### Before You Dive In

1. **Check existing issues** or open a new one to start a discussion
2. **Read [Chonkie's documentation](https://docs.chonkie.ai)** and core [concepts](https://docs.chonkie.ai/common/concepts)
3. **Set up your development environment** using the guide below

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/chonkie-inc/chonkie.git
cd chonkie

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (choose one)
pip install -e ".[dev]"                # Base development setup
pip install -e ".[dev,semantic]"       # If working on semantic features
pip install -e ".[dev,all]"            # For all features
```

## 🧪 Testing & Code Quality

### Running Tests

```bash
pytest                           # Run all tests
pytest tests/test_token_chunker.py    # Run specific test file
pytest --cov=chonkie            # Run tests with coverage
```

### Code Style

We use [ruff](https://github.com/astral-sh/ruff) for formatting and linting:

```bash
ruff check .                     # Check code quality
ruff check --fix .               # Auto-fix issues where possible
```

Our style configuration enforces:

- Code formatting (`F`)
- Import sorting (`I`) 
- Documentation style (`D`)
- Docstring coverage (`DOC`)

### Documentation Style

We follow Google-style docstrings:

```python
def chunk_text(text: str, chunk_size: int = 512) -> list[str]:
    """Split text into chunks of specified size.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If chunk_size <= 0
    """
    pass
```

## 📦 Project Structure

```
src/
├── chonkie/
    ├── chunker/     # Chunking implementations
    ├── embeddings/  # Embedding implementations
    └── refinery/    # Refinement utilities
```

## 🎯 Contribution Opportunities

### For Beginners

Start with issues labeled [`good-first-issue`](https://github.com/chonkie-inc/chonkie/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)

### Documentation

- Improve existing docs
- Add examples or tutorials
- Fix typos

### Code Improvements

- Implement new chunking strategies
- Add tokenizer support
- Optimize existing chunkers
- Improve test coverage

### Performance Enhancements

- Profile and optimize code
- Add benchmarks
- Improve memory usage

### New Features

Look for issues with [FEAT] labels, especially those from Chonkie Maintainers

## 🚦 Pull Request Process

### 1. Branch Naming

- `feature/description` for new features
- `fix/description` for bug fixes
- `docs/description` for documentation changes

### 2. Commit Messages

Write clear, descriptive commit messages:

```
feat: add batch processing to WordChunker

- Implement batch_process method
- Add tests for batch processing
- Update documentation
```

### 3. Dependencies

- Core dependencies go in `project.dependencies`
- Optional features go in `project.optional-dependencies`
- Development tools go in the `dev` optional dependency group

### 4. Code Review

- **Make sure your PR is for the `development` branch**
- All PRs need at least one review
- Maintainers will review for:
  - Code quality (via ruff)
  - Test coverage
  - Performance impact
  - Documentation completeness

## 🦛 Technical Details

### Semantic Versioning

Chonkie does not follow strict semantic versioning. We follow the following rules:

- 'MAJOR' version when we refactor/rewrite large parts of the codebase
- 'MINOR' version when we add breaking changes (e.g. changing a public API)
- 'PATCH' version when we add non-breaking features (e.g. adding a new chunker) or fix bugs

### Development Dependencies

Current development dependencies (as of April 9, 2025):

```toml
[project.optional-dependencies]
dev = [
    "tiktoken>=0.5.0",
    "datasets>=1.14.0",
    "transformers>=4.0.0",
    "pytest>=6.2.0", 
    "pytest-cov>=4.0.0",
    "pytest-xdist>=2.5.0",
    "coverage",
    "ruff>=0.0.265",
    "mypy>=1.11.0"
]
```

### Optional Dependencies

- `model2vec`: For model2vec embeddings
- `st`: For sentence-transformers
- `openai`: For OpenAI embeddings
- `cohere`: For Cohere embeddings
- `semantic`: For semantic features
- `all`: All optional dependencies

## 💡 Getting Help

- **Questions?** Open an issue or ask in Discord
- **Bugs?** Open an issue or report in Discord
- **Chat?** Join our Discord!
- **Email?** Contact [support@chonkie.ai](mailto:support@chonkie.ai)

## 🙏 Thank You

Every contribution helps make Chonkie better! We appreciate your time and effort in helping make Chonkie the CHONKiest it can be!

Remember:
> "A journey of a thousand CHONKs begins with a single commit" ~ Ancient Proverb, probably


## PR Checklist

- [ ] Branch name follows convention (feature/fix/docs: description)
- [ ] Commit messages are descriptive
- [ ] Tests pass locally
- [ ] Documentation updated if needed
