import pandas as pd
import wandb

api = wandb.Api()
entity, project = "fredella-pang-university-of-calgary-in-alberta","DST Continual Learning"
runs = api.runs(entity + "/" + project)

for run in runs:
    timestamp_list = []
    accuracy_list = []
    lr_list = []
    for i, row in run.history(samples=1000).iterrows():
        timestamp_list.append(row["_timestamp"])
        accuracy_list.append(row["test/acc"])
        lr_list.append(row["train/lr"])
    df = pd.DataFrame({"accuracy": accuracy_list})
    # df = pd.DataFrame({"accuracy": accuracy_list, "lr":lr_list})
    # df = pd.DataFrame({"timestamp": timestamp_list, "accuracy": accuracy_list})
    df.to_csv(f"csv/{run.config['sparsifier']}/{run.name}_sparsifier_{run.config['sparsifier']}_seed{run.config['seed']}.csv")
print("done")


    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
#     summary_list.append(run.summary._json_dict)

#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

#     # .name is the human-readable name of the run.
#     name_list.append(run.name)

# runs_df = pd.DataFrame(
#     {"summary": summary_list, "config": config_list, "name": name_list}
# )

# runs_df.to_csv("project.csv")    