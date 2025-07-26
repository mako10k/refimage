# RefImage Code Quality Automation
# This Makefile provides convenient commands for code quality checks

.PHONY: help install-dev format lint type-check security duplication quality-check quality-fix clean

# Default target
help:
	@echo "RefImage Code Quality Automation"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  help           - Show this help message"
	@echo "  install-dev    - Install development dependencies"
	@echo "  format         - Format code with Black and isort"
	@echo "  lint           - Run Flake8 linter"
	@echo "  type-check     - Run MyPy type checker"
	@echo "  security       - Run Bandit security scanner"
	@echo "  duplication    - Check code duplication with jscpd"
	@echo "  quality-check  - Run all quality checks (no fixes)"
	@echo "  quality-fix    - Run all quality checks with auto-fix"
	@echo "  clean          - Clean up generated files"

# Install development dependencies
install-dev:
	@echo "📦 Installing development dependencies..."
	pip install -e ".[dev]"
	npm install -g jscpd

# Format code
format:
	@echo "🎨 Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatting completed"

# Run linter
lint:
	@echo "🔍 Running Flake8 linter..."
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503,E501 --exclude=__pycache__,.git,.venv,venv

# Run type checker
type-check:
	@echo "🔍 Running MyPy type checker..."
	mypy src/refimage --ignore-missing-imports --no-strict-optional --warn-return-any --warn-unused-configs

# Run security scanner
security:
	@echo "🔍 Running Bandit security scanner..."
	bandit -r src/ -f json --severity-level medium

# Check code duplication
duplication:
	@echo "🔍 Checking code duplication..."
	jscpd --min-lines 5 --min-tokens 50 --reporters console,html --output ./jscpd-report src/ tests/

# Run all quality checks (no auto-fix)
quality-check:
	@echo "🚀 Running comprehensive code quality checks..."
	python scripts/quality_check.py

# Run all quality checks with auto-fix
quality-fix:
	@echo "🚀 Running comprehensive code quality checks with auto-fix..."
	python scripts/quality_check.py --fix

# Clean up generated files
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf jscpd-report/
	rm -f code-quality-report.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	@echo "✅ Cleanup completed"
