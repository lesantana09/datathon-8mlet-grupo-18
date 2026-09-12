.DEFAULT_GOAL := help

.PHONY: help sync-context test pre-commit run publish-policy

help: ## Show available commands
	@echo "Available commands:"
	@echo ""
	@echo "  make help             Show available commands"
	@echo "  make sync-context     Synchronize AI context (.ai/ai-context.md -> AGENTS.md, CLAUDE.md)"
	@echo "  make test             Run the test suite (uv run pytest)"
	@echo "  make pre-commit       Run pre-commit hooks on all files (Ruff + tests, same as CI)"
	@echo "  make run              Start API + MLflow (docker compose up -d)"
	@echo "  make publish-policy   Train the channel policy and publish it to MLflow (requires MLflow running)"

sync-context: ## Synchronize shared AI context into AGENTS.md and CLAUDE.md
	@bash .ai/sync-context.sh

test: ## Run the test suite
	@uv run pytest

pre-commit: ## Run pre-commit hooks on all files
	@uv run pre-commit run --all-files

run: ## Start API + MLflow
	@docker compose up -d

publish-policy: ## Train the channel policy and publish it to MLflow
	@uv run python -m datathon_mlet.train_and_publish_policy
