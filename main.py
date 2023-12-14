"""Main code"""
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
from BGWO1 import BGWO1
from BGWO2 import BGWO2

# ---Input-----------------------------------------------------------
#  feat     : feature vector (instances x features)
#  label    : label vector (instances x 1)
#  N        : Number of wolves
#  max_Iter : Maximum number of iterations

# ---Output----------------------------------------------------------
# sFeat    : Selected features (instances x features)
# Sf       : Selected feature index
# Nf       : Number of selected features
# curve    : Convergence curve
# -------------------------------------------------------------------


# fetch dataset
ionosphere = fetch_ucirepo(id=52)
# data (as pandas dataframes)
feat = ionosphere.data.features
label = ionosphere.data.targets
feat_as_list = np.array(feat.values.tolist())
dummy_label = np.array([0 if letter == 'b' else 1
                        if letter == 'g' else letter for letter in label.Class.values])
dummy = dummy_label.reshape(-1, 1)

# Set 20% data as validation set
HO = 0.2

# # Hold-out method
feat_train, feat_val, label_train, label_val = train_test_split(feat_as_list, dummy,
                                                                test_size=HO, random_state=1, stratify=None)

# Parameter setting
N = 10
MAX_ITER = 500
REPS = 31
bgwo1 = BGWO1(max_efos = MAX_ITER)
bgwo2 = BGWO2(max_efos = MAX_ITER)
algorithms = [bgwo1
              ,bgwo2
              ]
name_problem = ["BGWO1", "BGWO2"]
NUM_ALG = 0

df = pd.DataFrame({'Average Fitness': pd.Series(dtype='float'),
                   'Standard Deviation': pd.Series(dtype='float'),
                   'Best Fitness': pd.Series(dtype='float'),
                   'Worst Fitness': pd.Series(dtype='float'),
                   'Execution Time': pd.Series(dtype='float')
                   })
df2 = df
start_timer_p = time.time()

avg_curve_alg = []
best_avg_fitness_alg = []
best_std_fitness_alg = []
best_fitness_along_seeds = []
worst_fitness_along_seeds = []
alg_avg_time = []
total_time = []

for alg in algorithms:
    avg_curve = np.zeros(MAX_ITER, float)
    best_fitnes = np.zeros(REPS, float)
    time_by_repetition = np.zeros(REPS, float)
    print(alg)
    for s in range(0, REPS):
        start_timer = time.time()
        sFeat, Sf, Nf, curve = alg.evolve(feat_as_list, N, MAX_ITER, feat_train,
                             feat_val, label_train, label_val,seed=s)
        end_timer = time.time()
        print(s)
        time_spend = end_timer - start_timer
        avg_curve = avg_curve + curve
        time_by_repetition[s] = time_spend
        best_fitnes[s] = curve[MAX_ITER-1]
    avg_curve = avg_curve / REPS
    avg_best_fitnes = np.average(best_fitnes)
    std_best_fitnes = np.std(best_fitnes)
    avg_time = np.average(time_by_repetition)

    avg_curve_alg.append(avg_curve)
    best_avg_fitness_alg.append(avg_best_fitnes)
    best_std_fitness_alg.append(std_best_fitnes)
    best_fitness_along_seeds.append(min(best_fitnes))
    worst_fitness_along_seeds.append(max(best_fitnes))
    alg_avg_time.append(avg_time)
    total_time.append(math.fsum(time_by_repetition))
    # Plot convergence curve
    plt.plot(np.arange(1, MAX_ITER+1), curve)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Fitness Value')
    plt.title(str(name_problem[NUM_ALG]))
    plt.grid(True)
    plt.show()
    NUM_ALG = NUM_ALG + 1

new_row = pd.DataFrame({'Average Fitness': str(best_avg_fitness_alg[0]),
                        'Standard Deviation': str(best_std_fitness_alg[0]),
                        'Best Fitness': str(best_fitness_along_seeds[0]),
                        'Worst Fitness': str(worst_fitness_along_seeds[0]),
                        'Execution Time': str(alg_avg_time[0])}, index=[0])

new_row2 = pd.DataFrame({'Average Fitness': str(best_avg_fitness_alg[1]),
                        'Standard Deviation': str(best_std_fitness_alg[1]),
                        'Best Fitness': str(best_fitness_along_seeds[1]),
                        'Worst Fitness': str(worst_fitness_along_seeds[1]),
                        'Execution Time': str(alg_avg_time[1])}, index=[0])

df = pd.concat([df.loc[:], new_row]).reset_index(drop=True)
df2 = pd.concat([df2.loc[:], new_row2]).reset_index(drop=True)

df.to_csv("Binary-Greywolf-Optimization-Method-1-" + str(MAX_ITER) + ".csv", index=False)
df2.to_csv("Binary-Greywolf-Optimization-Method-2-" + str(MAX_ITER) + ".csv", index=False)
