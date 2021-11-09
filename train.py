import logging

import hydra
import pandas as pd

import src.utils as utils
from src.get_score import record_result
from src.load_data import InputData
from src.train_fold import train_fold

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    log.info("Started.")

    utils.fix_seed(c.params.seed)
    utils.debug_settings(c)
    device = utils.gpu_settings(c)

    run = utils.setup_wandb(c)

    input = InputData(c)

    oof_df = pd.DataFrame()
    losses = utils.AverageMeter()
    for fold in range(c.params.n_fold):
        log.info(f"========== fold {fold} training ==========")
        utils.fix_seed(c.params.seed + fold)

        _oof_df, score, loss = train_fold(c, input.train, fold, device)
        oof_df = pd.concat([oof_df, _oof_df])
        losses.update(loss)

        log.info(f"========== fold {fold} result ==========")
        record_result(c, _oof_df, fold, loss)

        if c.settings.debug:
            break

    oof_df.to_csv("oof_df.csv", index=False)

    log.info(f"========== final result ==========")
    score = record_result(c, oof_df, c.params.n_fold, losses.avg)

    log.info("Done.")

    utils.teardown_wandb(c, run, losses.avg)
    utils.send_result_to_slack(c, score, losses.avg)


if __name__ == "__main__":
    main()
