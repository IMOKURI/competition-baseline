.PHONY: help
.DEFAULT_GOAL := train

NOW = $(shell date '+%Y%m%d-%H%M%S')

train: ## Run training
	@nohup python train.py > /tmp/nohup_$(NOW).log &

tuning: ## Run training with parameter tuning
	@nohup python train.py --multirun > /tmp/nohup_$(NOW).log &


debug_train: ## Run training with debug
	@python train.py settings.debug=True hydra.verbose=True

debug_tuning: ## Run training with parameter tuning with debug
	@python train.py --multirun settings.debug=True hydra.verbose=True hydra.sweeper.n_trials=2


clean: ## Clean work directory
	@rm -rf ../outputs ../multirun


help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[38;2;98;209;150m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
