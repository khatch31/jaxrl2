import os

from download_wandb import export_to_csv

# SAVEDIR = "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/csv_files"
# PROJECT = "modelfree_finetuning_baselines2"
# ENTITY = "iris_intel"
# TASKS = ["kitchen_microwave+kettle+light switch+slide cabinet", "kitchen_microwave+kettle+bottom burner+light switch"]
# ALGORITHMS_DESCRIPTIONS = {"CQL":["default", "cql_alpha0"], "IQL":["default"]}
# SEEDS = [0, 1, 2]

SAVEDIR = "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/csv_files"
PROJECT = "MW10_2_mf_baselines"
ENTITY = "iris_intel"
TASKS = ["metaworld_assembly-v2",
         "metaworld_bin-picking-v2",
         "metaworld_box-close-v2",
         "metaworld_coffee-push-v2",
         "metaworld_disassemble-v2",
         "metaworld_door-open-v2",
         "metaworld_drawer-open-v2",
         "metaworld_hammer-v2",
         "metaworld_plate-slide-v2",
         "metaworld_window-open-v2",]

ALGORITHMS_DESCRIPTIONS = {"CQL":["noproprio", "cql_alpha0_noproprio"], "IQL":["noproprio"]}
SEEDS = [0, 1, 2]

def download_many(savedir, project, entity, tasks, algorithms_descriptions, seeds):
    for task in tasks:
        for algorithm, descriptions in algorithms_descriptions.items():
            for description in descriptions:
                for seed in seeds:
                    try:
                        export_to_csv(savedir, entity, project, task, description, algorithm, seed)
                    except:
                        print(f"Skipped {os.path.join(savedir, entity, project, task, description, algorithm, f'seed_{seed}')}")

    # import os
    # import numpy as np
    # folders = ["/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/q_visualization/CVMVE_theory",
    #            "/iris/u/khatch/vd5rl/jaxrl2/baselines_finetuning/q_visualization/OOD3"]
    #
    # for folder in folders:
    #     cql = np.loadtxt(os.path.join(folder, "q_viz_cql.csv"), delimiter=",").T
    #     iql = np.loadtxt(os.path.join(folder, "q_viz_iql.csv"), delimiter=",").T
    #     cql0 = np.loadtxt(os.path.join(folder, "q_viz_cql0.csv"), delimiter=",").T
    #
    #     header_cql = ",".join([f"cql_ep_{i}" for i in range(cql.shape[1])])
    #     header_iql = ",".join([f"iql_ep_{i}" for i in range(iql.shape[1])])
    #     header_cql0 = ",".join([f"cql0_ep_{i}" for i in range(cql0.shape[1])])
    #     header = header_cql + "," + header_iql + "," + header_cql0
    #
    #     data = np.concatenate((cql, iql, cql0), axis=1)
    #     np.savetxt(os.path.join(folder, "cql_iql_cql0.csv"), data, delimiter=",")




if __name__ == "__main__":
    download_many(SAVEDIR, PROJECT, ENTITY, TASKS, ALGORITHMS_DESCRIPTIONS, SEEDS)
