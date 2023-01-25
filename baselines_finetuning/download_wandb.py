import os
import wandb

def export_to_csv(csv_path, entity, project, task, description, algorithm, seed):
    group_name = f"{task}_{algorithm}_{description}"
    name = f"seed_{seed}"
    run_id=group_name + "-" + name

    api = wandb.Api()
    print(f"\nDownloading wandb data for {entity}/{project}/{run_id}")
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()

    print("Saving wandb data...")
    csv_file = os.path.join(csv_path, entity, project, task, description, algorithm, f"seed_{seed}.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    history.to_csv(path_or_buf=csv_file)
    print(f"Saved to \"{csv_file}\".")

def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Render Adoit Hand")
    parser.add_argument("--savedir", type=str,  help="")
    parser.add_argument("--entity", type=str,  help="")
    parser.add_argument("--project", type=str,  help="")
    parser.add_argument("--task", type=str,  help="")
    parser.add_argument("--description", type=str,  help="")
    parser.add_argument("--algorithm", type=str,  help="")
    parser.add_argument("--seed", type=int,  help="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    export_to_csv(args.savedir, args.entity, args.project, args.task, args.description, args.algorithm, args.seed)

"""
python3 -u download_wandb.py \
--savedir "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/csv_files" \
--entity iris_intel \
--project modelfree_finetuning_baselines2 \
--task "kitchen_microwave+kettle+light switch+slide cabinet" \
--description default \
--algorithm CQL \
--seed 0
"""
