.PHONY: help, ci-black, ci-flake8, ci-test, isort, black, docs, dev-start, dev-stop

PROJECT=mlfaker
CONTAINER_NAME="mlfaker_bash_${USER}"  ## Ensure this is the same name as in docker-compose.yml file
PROJ_DIR="/mnt/mlfaker"
VERSION_FILE:=VERSION
TAG:=$(shell cat ${VERSION_FILE})

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

git-tag:  ## Tag in git, then push tag up to origin
	git tag $(TAG)
	git push origin $(TAG)

ci-black: dev-start ## Test lint compliance using black. Config in pyproject.toml file
	docker exec -t $(CONTAINER_NAME) black --check $(PROJ_DIR)

ci-flake8: dev-start ## Test lint compliance using flake8. Config in tox.ini file
	docker exec -t $(CONTAINER_NAME) flake8 $(PROJ_DIR)

ci-test: dev-start ## Runs unit tests using pytest
	docker exec -t $(CONTAINER_NAME) pytest $(PROJ_DIR)

ci-mypy: dev-start ## Runs mypy type checker
	docker exec -t $(CONTAINER_NAME) mypy --ignore-missing-imports $(PROJ_DIR)

ci-test-interactive:  ## Runs unit tests using pytest, and gives you an interactive IPDB session at the first failure
	docker exec -it $(CONTAINER_NAME) pytest $(PROJ_DIR)  -x --pdb --pdbcls=IPython.terminal.debugger:Pdb

ci: ci-black ci-flake8 ci-test ci-mypy ## Check black, flake8, and run unit tests
	@echo "CI successful"

isort: dev-start  ## Runs isort to sorts imports
	docker exec -t $(CONTAINER_NAME) isort $(PROJ_DIR)

black: dev-start ## Runs black auto-linter
	docker exec -t $(CONTAINER_NAME) black $(PROJ_DIR)

lint: format ## Deprecated. Here to support old workflow

format: isort black ## Formats repo; runs black and isort on all files
	@echo "Formatting complete"

dev-start: ## Primary make command for devs, spins up containers
	docker-compose -f docker-compose.yml --project-name $(PROJECT) up -d --no-recreate

dev-stop: ## Spin down active containers
	docker-compose -f docker-compose.yml --project-name $(PROJECT) down

dev-rebuild: ## Rebuild images for dev containers (useful when Dockerfile/requirements are updated)
	docker-compose -f docker-compose.yml --project-name $(PROJECT) up -d --build

bash: dev-start ## Exec into docker bash terminal
	docker exec -it $(CONTAINER_NAME) bash

docs: ## Build docs using Sphinx and copy to docs folder (this makes it easy to publish to gh-pages)
	docker exec -e GRANT_SUDO=yes $(CONTAINER_NAME) bash -c "cd docsrc; make html"
	@cp -a docsrc/_build/html/. docs

ipython: ## Provides an interactive ipython prompt
	docker exec -it $(CONTAINER_NAME) ipython

clean: ## Clean out temp/compiled python files
	find . -name __pycache__ -delete
	find . -name "*.pyc" -delete
