.DEFAULT_GOAL := help

.PHONY: help sync-context

help: ## Show available commands
	@echo "Available commands:"
	@echo ""
	@echo "  make help           Show available commands"
	@echo "  make sync-context   Synchronize AI context (.ai/ai-context.md -> AGENTS.md, CLAUDE.md)"

sync-context: ## Synchronize shared AI context into AGENTS.md and CLAUDE.md
	@bash .ai/sync-context.sh
