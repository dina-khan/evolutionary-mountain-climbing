import os
import glob
import numpy as np
import pandas as pd

# helper function which calculates the mean and standard deviation 
# of input values and represents it as a string 
def get_mean_std(v):
    return f"{float(np.mean(v)):.3f}, std:  {float(np.std(v)):.3f}"

# This function returns a summary dictionary for the given experiment folder,
# aggregating all runs of that experiment
def summarize_experiment(experiment_folder):

    # getting all experiment run folders within the experiment folder
    run_folders = sorted(glob.glob(os.path.join(experiment_folder, "run*")))

    # The table will include last generation elites minimum mountain radius reached, 
    # and improvement over generations in minimum mountain radius reached, 
    # aggregated across runs for each experiment
    final_elite_radius = []
    improvement_elite_radius = []
    
    # Iterating over each experiment run folder within the experiment folder
    for run_folder in run_folders:
        # storing the csv metrics as a data frame
        df = pd.read_csv(os.path.join(run_folder, "metrics.csv"))

        # Storing generations as an array as it will be used as the x axis
        # to calculate area under the curve for generalized metrics
        generations = df["generation"].to_numpy(dtype=float)

        # The elite minimum horizontal mountain radius reached last generation 
        # values across are taken from the last row of the datafram 
        final_elite_radius.append(float(df.iloc[-1]["elite_min_radius"]))

        # Storing the elite creature's minimum radius 
        # from the mountain across generations as a dataframe
        elite_min_radius_generations = df["elite_min_radius"].to_numpy(dtype=float)

        # Subtracting the minimum radius of elite creatures in subsequent generations 
        # from the minimum radius of the first generation elite gives improvement values
        # in minimum radius over generations
        elite_min_radius_improvement = elite_min_radius_generations[0] - elite_min_radius_generations

        # Using area under the curve to integrate the minimum radius improvement 
        # values over generations and get a single value to represent improvement
        elite_min_radius_improvement_over_generations = float(np.trapezoid(elite_min_radius_improvement, generations))

        # Dividing improvement in minimum radius over the number of generations 
        # to get mean improvement in minimum radius over generations
        improvement_elite_radius.append(elite_min_radius_improvement_over_generations / float(generations[-1]))

        # storing the experiment name
        experiment_name = os.path.basename(os.path.normpath(experiment_folder))

    # returning calculated dictionary summary for given experiment 
    return {
        "Experiment": experiment_name,
        "Last generation elite minimum radius": get_mean_std(final_elite_radius),
        "Improvement in elite minimum radius over generations": get_mean_std(improvement_elite_radius),
    }
# Prints summary for every experiment folder inside result logs directory
def start_summary():
    # Getting all experiment folders from result logs directory
    experiment_folders = sorted([e for e in glob.glob(os.path.join("result_logs", "*")) if os.path.isdir(e)])

    # This list will store summary dictionaries for each experiment folder
    experiment_summaries = []

    # For each experiment folder, adding a summary dictionary 
    for e in experiment_folders:
        experiment_summaries.append(summarize_experiment(e))

    # Making list of experiment summaries into data frame and saving it as a csv
    experiment_summaries_df = pd.DataFrame(experiment_summaries)
    experiment_summaries_df.to_excel("summary.xlsx", index=False)

if __name__ == "__main__":
    start_summary()

