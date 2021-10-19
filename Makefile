.PHONY: help
.DEFAULT_GOAL := train

NOW = $(shell date '+%Y%m%d-%H%M%S')

train: ## Run training
	@nohup python train.py > /tmp/nohup_$(NOW).log &


debug_train: ## Run training with debug
	@python train.py settings.debug=True


clean: ## Clean work directory
	@rm -rf outputs


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
