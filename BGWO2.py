"""Algoritmo BGWO2"""
import math
import numpy as np
from FitnessFunction import fitness_function

class BGWO2:
    def __init__(self, max_efos: int):
        self.max_efos = max_efos

    def evolve(self, feat, wolfs, max_iter, feat_train, feat_val, label_train, label_val, seed:int):
        """Función BGWO1"""
        np.random.seed(seed)
        dim = feat.shape[1]
        population = np.zeros((wolfs, dim))
        for i in range(wolfs):
            for d in range(dim):
                if np.random.rand() > 0.5:
                    population[i, d] = 1
        fit = np.zeros(wolfs)
        for i in range(wolfs):
            fit[i] = fitness_function(population[i,:],feat_train, feat_val, label_train, label_val)
        idx = np.argsort(fit)
        # Definition of Xp(t)
        x_alpha = population[idx[0]]
        x_beta = population[idx[1]]
        x_delta = population[idx[2]]
        # Definition of Xp(t) Fitness
        f_alpha = fit[idx[0]]
        f_beta = fit[idx[1]]
        f_delta = fit[idx[2]]
        curve = []
        t = 1
        # rest of the code

        # ---Iterations start-----------------------------------------------

        while t <= max_iter:
            # Equation 17 ==> a = 2 - 2 * (t / max_iter)
            a = 2 - 2 * (t / max_iter)
            for i in range(wolfs):
                for d in range(dim):
                    # Equation 16 ==> C = 2 * r2
                    c1 = 2 * np.random.rand()
                    c2 = 2 * np.random.rand()
                    c3 = 2 * np.random.rand()
                    # Equation (14) 22, 23 and 24 ==> D = |C * Xp(t) - X(t)|
                    d_alpha = abs(c1 * x_alpha[d] - population[i][d])
                    d_beta = abs(c2 * x_beta[d] - population[i][d])
                    d_delta = abs(c3 * x_delta[d] - population[i][d])
                    # Equation 15 ==> A = 2 * a * r1 - a
                    a1 = 2 * a * np.random.rand() - a
                    a2 = 2 * a * np.random.rand() - a
                    a3 = 2 * a * np.random.rand() - a
                    x1 = x_alpha[d] - a1 * d_alpha
                    x2 = x_alpha[d] - a2 * d_beta
                    x3 = x_alpha[d] - a3 * d_delta
                    xn = (x1 + x2 + x3) / 3
                    tf = 1 / (1 + math.exp(-10 * (xn - 0.5)))
                    if tf >= np.random.rand():
                        population[i][d] = 1
                    else:
                        population[i][d] = 0
            for i in range(wolfs):
                fit[i] = fitness_function(population[i, :], feat_train,
                                          feat_val, label_train, label_val)
                if fit[i] < f_alpha:
                    f_alpha = fit[i]
                    x_alpha = population[i, :]
                if fit[i] < f_beta and fit[i] > f_alpha:
                    f_beta = fit[i]
                    x_beta = population[i, :]
                if fit[i] < f_delta and fit[i] > f_alpha and fit[i] > f_beta:
                    f_delta = fit[i]
                    x_delta = population[i, :]
            curve.append(f_alpha)
            t = t + 1
        sf = [i for i, x in enumerate(x_alpha) if x == 1]
        nf = len(sf)
        sfeat = feat[:, sf]

        return sfeat, sf, nf, curve
