import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import wandb
from icecream import ic

plt.style.use("science")

sweep_gpt = pd.read_csv("sweep_gpt.csv")
sweep_ngpt = pd.read_csv("sweep_ngpt.csv")

# sort by best_eval_loss
sweep_gpt = sweep_gpt.sort_values("best_eval_loss")
sweep_ngpt = sweep_ngpt.sort_values("best_eval_loss")
ic(sweep_gpt, sweep_ngpt)

# print the run_ids of the best 6 runs
ic("|".join(sweep_gpt.head(6)["run_id"].tolist()), "|".join(sweep_ngpt.head(6)["run_id"].tolist()))


# find in each the row with the best best_eval_loss
best_gpt = sweep_gpt[sweep_gpt["best_eval_loss"] == sweep_gpt["best_eval_loss"].min()]
best_ngpt = sweep_ngpt[sweep_ngpt["best_eval_loss"] == sweep_ngpt["best_eval_loss"].min()]
# convert to dicts
best_gpt = best_gpt.to_dict(orient="records")[0]
best_ngpt = best_ngpt.to_dict(orient="records")[0]
ic(best_gpt, best_ngpt)

api = wandb.Api()
best_gpt_run = api.run(f"/tudoroancea/tadam/runs/{best_gpt['run_id']}")
best_ngpt_run = api.run(f"/tudoroancea/tadam/runs/{best_ngpt['run_id']}")

best_gpt_hist: pd.DataFrame = best_gpt_run.history(samples=1100)
best_ngpt_hist: pd.DataFrame = best_ngpt_run.history(samples=1100)
eval_interval = 200

best_gpt_eval_loss = best_gpt_hist["eval_loss"].dropna().tolist()
best_gpt_train_loss = best_gpt_hist["train_loss"].dropna().tolist()
best_ngpt_eval_loss = best_ngpt_hist["eval_loss"].dropna().tolist()
best_ngpt_train_loss = best_ngpt_hist["train_loss"].dropna().tolist()

plt.figure(figsize=(6, 4))
plt.plot(eval_interval * np.arange(len(best_gpt_eval_loss)), best_gpt_eval_loss, "r:", label="GPT evaluation loss")
plt.plot(np.arange(len(best_gpt_train_loss)), best_gpt_train_loss, "r-", label="GPT training loss")
plt.plot(eval_interval * np.arange(len(best_ngpt_eval_loss)), best_ngpt_eval_loss, "b:", label="nGPT evaluation loss")
plt.plot(np.arange(len(best_ngpt_train_loss)), best_ngpt_train_loss, "b-", label="nGPT training loss")
plt.ylabel("loss")
plt.xlabel("training steps")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("best_runs.svg")
# plt.show()


# nhow create two new plots: one with the best run for each model size of GPT, the other of nGPT
sizes = ["small", "medium", "large"]
gpt_by_size = {size: sweep_gpt[sweep_gpt["model_size"] == size].copy() for size in sizes}
best_gpt = {
    size: gpt_by_size[size][gpt_by_size[size]["best_eval_loss"] == gpt_by_size[size]["best_eval_loss"].min()].to_dict(
        orient="records"
    )[0]
    for size in sizes
}
plt.figure(figsize=(6, 5))
for size, color in zip(sizes, ["r", "b", "g"], strict=True):
    ic(best_gpt[size]["run_id"])
    best_gpt_hist = api.run(f"/tudoroancea/tadam/runs/{best_gpt[size]['run_id']}").history(samples=1100)
    best_gpt_eval_loss = best_gpt_hist["eval_loss"].dropna().tolist()
    best_gpt_train_loss = best_gpt_hist["train_loss"].dropna().tolist()
    plt.plot(
        eval_interval * np.arange(len(best_gpt_eval_loss)),
        best_gpt_eval_loss,
        color + ":",
        label=f"GPT {size} evaluation loss",
    )
    plt.plot(
        np.arange(len(best_gpt_train_loss)),
        best_gpt_train_loss,
        color + "-",
        label=f"GPT {size} training loss",
    )

plt.ylabel("loss")
plt.xlabel("training steps")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("best_gpt.svg")

ngpt_by_size = {size: sweep_ngpt[sweep_ngpt["model_size"] == size].copy() for size in sizes}
best_ngpt = {
    size: ngpt_by_size[size][
        ngpt_by_size[size]["best_eval_loss"] == ngpt_by_size[size]["best_eval_loss"].min()
    ].to_dict(orient="records")[0]
    for size in sizes
}
plt.figure(figsize=(6, 5))
for size, color in zip(sizes, ["r", "b", "g"], strict=True):
    ic(best_ngpt[size]["run_id"])
    best_ngpt_hist = api.run(f"/tudoroancea/tadam/runs/{best_ngpt[size]['run_id']}").history(samples=1100)
    best_ngpt_eval_loss = best_ngpt_hist["eval_loss"].dropna().tolist()
    best_ngpt_train_loss = best_ngpt_hist["train_loss"].dropna().tolist()
    plt.plot(
        eval_interval * np.arange(len(best_ngpt_eval_loss)),
        best_ngpt_eval_loss,
        color + ":",
        label=f"nGPT {size} evaluation loss",
    )
    plt.plot(
        np.arange(len(best_ngpt_train_loss)),
        best_ngpt_train_loss,
        color + "-",
        label=f"nGPT {size} training loss",
    )

plt.ylabel("loss")
plt.xlabel("training steps")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("best_ngpt.svg")
plt.show()
