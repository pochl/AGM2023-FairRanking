from logging import getLogger
from pathlib import Path
from time import time

from func import compute_pi_expo_fair
from func import compute_pi_max
from func import compute_pi_nsw
from func import compute_pi_nsw_lag1
from func import compute_pi_nsw_lag2
from func import compute_pi_unif
from func import evaluate_pi
from func import exam_func
from func import synthesize_rel_mat
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from pandas import DataFrame


logger = getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    logger.info(f"The current working directory is {Path().cwd()}")
    start_time = time()

    # configurations
    n_doc = cfg.setting.n_doc
    K_list = cfg.setting.K_list
    pol_list = ["unif", "max", "expo-fair", "nsw", "nsw_1"]
    result_df = DataFrame(
        columns=[
            "policy",
            "seed",
            "K",
            "user_util",
            "mean_max_envy",
            "pct_better_off",
            "pct_worse_off",
            "nsw",
        ]
    )

    # log path
    log_path = Path("./varying_K")
    log_path.mkdir(exist_ok=True, parents=True)

    elapsed_prev = 0.0
    for k in K_list:
        v = exam_func(n_doc, k, cfg.setting.exam_func)
        result_df_box = DataFrame(
            columns=["policy", "item_util", "max_envy", "imp_in_item_util"]
        )
        for s in np.arange(cfg.setting.n_seeds):
            rel_mat_true, rel_mat_obs = synthesize_rel_mat(
                n_query=cfg.setting.n_query,
                n_doc=n_doc,
                lam=cfg.setting.default_lam,
                noise=cfg.setting.default_noise,
                flip_ratio=cfg.setting.flip_ratio,
                random_state=s,
            )

            item_utils_unif = None
            user_util_list = []
            mean_max_envy_list = []
            pct_item_util_better_off_list = []
            pct_item_util_worse_off_list = []

            for pol in pol_list:

                if pol == "max":
                    opt = compute_pi_max
                elif pol == "expo-fair":
                    opt = compute_pi_expo_fair
                elif pol == "nsw":
                    opt = compute_pi_nsw
                elif pol == "unif":
                    opt = compute_pi_unif
                elif pol == "nsw_1":
                    opt = compute_pi_nsw_lag1
                elif pol == "nsw_2":
                    opt = compute_pi_nsw_lag2
                else:
                    raise ValueError("Invalid pol")
                
                pi = opt(rel_mat=rel_mat_obs, v=v)
                user_util, item_utils, max_envies, _ = evaluate_pi(
                    pi=pi,
                    rel_mat=rel_mat_true,
                    v=v,
                )

                if pol == "unif":
                    item_utils_unif = item_utils
                else:
                    result_pol = DataFrame(
                        data={
                            "policy": [pol] * n_doc,
                            "item_util": item_utils,
                            "max_envy": max_envies,
                            "imp_in_item_util": item_utils / item_utils_unif,
                        }
                    )

                    result_df_box = pd.concat([result_df_box, result_pol])

                user_util_list.append(user_util)
                mean_max_envy_list.append(max_envies.mean())
                pct_item_util_better_off_list.append(
                    100 * ((item_utils / item_utils_unif) > 1.10).mean()
                )
                pct_item_util_worse_off_list.append(
                    100 * ((item_utils / item_utils_unif) < 0.90).mean()
                )


            result_df_ = DataFrame(
                data={
                    "policy": pol_list,
                    "seed": [s] * len(pol_list),
                    "K": [k] * len(pol_list),
                    "user_util": user_util_list,
                    "mean_max_envy": mean_max_envy_list,
                    "pct_better_off": pct_item_util_better_off_list,
                    "pct_worse_off": pct_item_util_worse_off_list,
                },
            )
            result_df = pd.concat([result_df, result_df_])
        elapsed = np.round((time() - start_time) / 60, 2)
        diff = np.round(elapsed - elapsed_prev, 2)
        logger.info(f"k={k}: {elapsed}min (diff {diff}min)")
        elapsed_prev = elapsed

        result_df_box.reset_index(inplace=True)
        result_df_box.to_csv(log_path / "result_df_box.csv")

    result_df.reset_index(inplace=True)
    result_df.to_csv(log_path / "result_df.csv")


if __name__ == "__main__":
    main()
