.DEFAULT_GOAL := help

.PHONY: help dev local experiment aws-init aws-plan aws-apply aws-destroy aws-output sync train test pre-commit up down start stop destroy rebuild rebuild-all logs shell publish

help: ## Show available commands
	@echo	Datathon MLET - Grupo 18
	@echo	---
	@echo Available commands:
	@echo - make help         Show available commands
	@echo - make dev          Creates the development environment locally.
	@echo - make local        Runs the API locally.
	@echo - make experiment   Runs the Experiments and Artifacts Generation.
	@echo - make aws-init     Initializes the AWS environment (terraform init).
	@echo - make aws-plan     Plans the AWS environment (terraform plan).
	@echo - make aws-apply    Applies the AWS environment (terraform apply).
	@echo - make aws-destroy  Destroys the AWS environment (terraform destroy).
	@echo - make aws-output   Outputs the AWS environment (terraform output).
	@echo - make dataset      Downloads the dataset from Kaggle.
	@echo - make sync         Runs the Data Lake Sync.
	@echo - make train        Runs the Training Pipeline.
	@echo - make test         Run the test suite (uv run pytest)
	@echo - make format       Runs the code formatter (uv run ruff check --fix .)
	@echo - make pre-commit   Run pre-commit hooks on all files (Ruff + tests, same as CI)
	@echo - make up           Builds, creates, starts, and attaches to containers for API + Observability tools (docker compose up -d)
	@echo - make down         Stop and remove containers, networks, volumes, and images created for API + Observability tools (docker compose down)
	@echo - make start        Starts API + Observability tools (docker compose start)
	@echo - make stop         Stops API + Observability tools (docker compose stop)
	@echo - make publish      Publish the API image to Docker Hub (docker compose push api)
	@echo - make destroy      Stop and remove API + Observability tools (docker compose down -v)
	@echo - make rebuild      Rebuild and force restart the API service (docker compose up -d --build --force-recreate api)
	@echo - make rebuild-all  Rebuild and force restart all services (docker compose up -d --build --force-recreate)
	@echo - make logs         Show logs of all services (docker compose logs -n 50)
	@echo - make shell        Run a command in the API container (docker compose exec api bash)


dev: ## Creates the development environment locally
	@uv sync --extra dev

local: ## Runs the API locally
	@uv run uvicorn api.main:app --reload --port 8081

experiment: ## Runs the Experiments and Artifacts Generation
	@(uv run python -c "\
from datathon_mlet.data_prep import load_clean_dataset, prepare_features\
from datathon_mlet.experiments import log_baseline_vs_thompson_sampling\
from pathlib import Path\
df = load_clean_dataset(Path('data/processed/bank_marketing_clean.parquet'))\
prep = prepare_features(df)\
log_baseline_vs_thompson_sampling(prep.action, prep.reward, arms=['cellular', 'telephone'], baseline_arm='telephone', n_seeds=20)\
"\
	)

aws-init: ## Initializes the AWS environment
	@terraform -chdir=deploy/terraform init

aws-plan: ## Plans the AWS environment
	@terraform -chdir=deploy/terraform plan

aws-apply: ## Applies the AWS environment
	@terraform -chdir=deploy/terraform apply

aws-destroy: ## Destroys the AWS environment
	@terraform -chdir=deploy/terraform destroy

aws-output: ## Outputs the AWS environment
	@terraform -chdir=deploy/terraform output

dataset: ## Downloads the dataset from Kaggle
	@uv run pip install kaggle
	@kaggle datasets download -d henriqueyamahata/bank-marketing

sync: ## Runs the Data Lake Sync
	@uv run python -m datathon_mlet.data_lake_sync

train: ## Runs the Training Pipeline
	@uv run python -m datathon_mlet.train_and_publish_policy

test: ## Run the test suite
	@uv run pytest

format: ## Run the code formatter (uv run ruff check --fix .)
	@uv run ruff check --fix .

pre-commit: ## Run pre-commit hooks on all files (Ruff + tests, same as CI)
	@uv run pre-commit run --all-files

up: ## Builds, creates, starts, and attaches to containers for API + Observability tools
	@docker compose up -d

down: ## Stop and remove containers, networks, volumes, and images created for API + Observability tools
	@docker compose down

start: ## Start API + Observability tools
	@docker compose start

stop: ## Stop API + Observability tools
	@docker compose stop

publish: ## Publish the API image to Docker Hub
	@docker compose push api

destroy: ## Stop and remove API + Observability tools
	@docker compose down -v

rebuild: ## Rebuild and force restart the API service
	@docker compose up -d --build --force-recreate api

rebuild-all: ## Rebuild and force restart all services
	@docker compose up -d --build --force-recreate

logs: ## Show logs of all services
	@docker compose logs -n 50

shell: ## Run a command in the API container
	@docker compose exec api bash
