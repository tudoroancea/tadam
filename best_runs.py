import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import scienceplots  # noqa: F401
# plt.style.use("science")
from icecream import ic

import wandb


sweep_gpt = pd.read_csv("sweep_gpt.csv")
sweep_ngpt = pd.read_csv("sweep_ngpt.csv")

# find in each the row with the best best_eval_loss
best_gpt = sweep_gpt[sweep_gpt["best_eval_loss"] == sweep_gpt["best_eval_loss"].max()]
best_ngpt = sweep_ngpt[sweep_ngpt["best_eval_loss"] == sweep_ngpt["best_eval_loss"].max()]
ic(best_gpt, best_ngpt)

api = wandb.Api()
best_gpt_run = api.run(f"/tudoroancea/tadam/runs/{best_gpt['run_id'].iloc[0]}")
# best_ngpt_run = api.run(f"/tudoroancea/tadam/runs/{best_ngpt['run_id'].iloc[0]}")

best_gpt_hist: pd.DataFrame = best_gpt_run.history(samples=5500)
# best_ngpt_hist: pd.DataFrame = best_ngpt_run.history(samples=5500)
ic(best_gpt_hist)
eval_interval = 200
# plot the eval loss and train loss, skipping
best_gpt_eval_loss = best_gpt_hist["eval_loss"].dropna().tolist()
best_gpt_train_loss = best_gpt_hist["train_loss"].dropna().tolist()
plt.plot(eval_interval * np.arange(len(best_gpt_eval_loss)), best_gpt_eval_loss, label="eval loss")
plt.plot(np.arange(len(best_gpt_train_loss)), best_gpt_train_loss, label="train loss")
plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig("best_gpt.svg")
plt.show()
