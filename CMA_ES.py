# -*- coding: utf-8 -*-
from deap import creator, base, tools, cma
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import numpy as np
import pandas as pd


# Chargement d'un set de données random
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)
X_train, X_test, y_train, y_test = train_test_split(X, y)

parametres = {
        "base_score" : [0.1, 0.9],
        "colsample_bylevel": [0.1, 0.9],
        "colsample_bytree": [0.1, 0.9],
        "gamma": [0, 0.9],
        "learning_rate": [0.1, 0.9], 
        "max_delta_step": [0, 0.9],
        "max_depth": [5,15],
        "n_estimators": [10, 1000],
        "reg_alpha": [0, 1],
        "reg_lambda": [0, 1],
        "scale_pos_weight": [0, 1], 
        "seed": [0, 10],
        "subsample": [0, 1],
        "max_leaves": [1, 1000]}

def models(individual):
    # splitting
    p = [ parametres[order[_]][0] + individual[_]*(parametres[order[_]][1]-parametres[order[_]][0]) for _ in range(len(order))]

    params =[]
    for element in p:
        params.append(max(0, element))

    regressor = xgb.XGBRegressor(
        base_score=params[0], 
        colsample_bylevel=min(1, params[1]), 
        colsample_bytree=min(1,params[2]),
        gamma=params[3], 
        learning_rate=params[4], 
        max_delta_step=params[5], 
        max_depth=int(params[6])+1,
        n_estimators=int(params[7]), 
        reg_alpha=params[8], 
        reg_lambda=params[9],
        scale_pos_weight=params[10], 
        seed=int(params[11]),  
        subsample=min(1, params[12]),
        max_leaves=int(params[13]))

    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    return err


order = list(parametres.keys())			   


centroide_initial = [0.5] * len(parametres.items())
initial_guess = np.array(centroide_initial)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", models)

def main():

    # Nombre max de générations
    NGENMAX = 100
    # Nombre d'individus dans la population
    NPOP = 12

    # Enregistrement de la stratégie
    strategy = cma.Strategy(centroide_initial, sigma=0.3, lambda_=NPOP)
    toolbox.register(u"generate", strategy.generate, creator.Individual)
    toolbox.register(u"update", strategy.update)

    import multiprocessing

    pool = multiprocessing.Pool(processes=6)
    toolbox.register("map", pool.map)
       
    # Continue on with the evolutionary algorithm
    # Statistiques
    hof   = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register(u"avg", np.mean, axis=0)
    stats.register(u"std", np.std, axis=0)
    stats.register(u"min", np.min, axis=0)
    stats.register(u"max", np.max, axis=0)
    hof_complet = np.zeros( (NGENMAX,len(centroide_initial)) )
    ecartypes   = np.zeros( (NGENMAX,len(centroide_initial)) )
    # Pour l'analyse d'identifiabilite
    toutlemonde = np.zeros( (NGENMAX, NPOP, len(centroide_initial)) )
    touslesfits = np.zeros( (NGENMAX, NPOP) )
    
    # Initialisation du logbook
    column_names = [u"gen", u"evals"]
    if stats is not None:
        column_names += stats.functions.keys()
    logbook = tools.Logbook()
    logbook.header = column_names
	
	 # Algorithme
    STOP = False
    gen = 0
    while not STOP:
        
        # Generate a new population
        population = toolbox.generate()
        
        # Ecartype/moyenne de chaque parametre de la population
        foo = np.array(population)
        ecartypes[gen] = np.abs( np.std(foo,axis=0) / np.mean(foo,axis=0) )
        toutlemonde[gen] = foo
        
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        k = 0
        for ind, fit in zip(population, fitnesses):
            #ind.fitness.values = fit
            touslesfits[gen, k] = fit#[0]
            k += 1
        
        # Update hall of fame
        hof.update(population)
        hof_complet[gen] = hof.items[0]
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        
        # Update statistics
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(population), **record)
        print(logbook.stream)
        
        gen +=1
        
        # On vérifie si on doit s'arrêter
        conv_paras   = np.max(ecartypes[gen-1]) < 1e-3

        conv_fitness = (np.array(fitnesses).std() / np.array(fitnesses).mean()) < 1e-5
        if (conv_paras and conv_fitness) or gen >= NGENMAX:
            STOP = True
    
    return logbook, ecartypes, toutlemonde, touslesfits, hof, hof_complet
    
if __name__ == "__main__":

 # print 'Erreur systématique : [%s]' % ', '.join(map(str, accuracy))
    logbook, ecartypes, toutlemonde, touslesfits, hof, hof_complet = main()
    # Mise en forme des résultats
    Ecartypes = ecartypes
    toutlemonde_2 = np.column_stack( [np.ravel(toutlemonde[:,:,_]) for _ in range(len(centroide_initial))] )
    OptGlobal = np.zeros_like(centroide_initial)
    OptParGen = np.zeros_like(hof_complet)
    ToutLeMonde = np.zeros_like(toutlemonde_2)
    for i in range(len(centroide_initial)):
        OptGlobal[i]     = parametres[order[i]][0] + hof.items[0][i]   *(parametres[order[i]][1]-parametres[order[i]][0])
        OptParGen[:,i]   = parametres[order[i]][0] + hof_complet[:,i]   *(parametres[order[i]][1]-parametres[order[i]][0])
        ToutLeMonde[:,i] = parametres[order[i]][0] + toutlemonde_2[:,i] *(parametres[order[i]][1]-parametres[order[i]][0])

   # On garde les statistiques intéressantes
    sigma_f = np.array( logbook.select(u"std") )
    minimum = np.array( logbook.select(u"min") )
    ngen = len(sigma_f)
    
    # Sauvegarde du meilleur individu et de l'ordre de ses propriétés
    #meilleur = np.column_stack((list( OptGlobal[_].values() ) for _ in NMATER))
    np.savetxt(u"CMA_bestfit.txt", OptGlobal)
    #np.savetxt('CMA_bestfit_tag.txt', list( OptGlobal[0].keys() ), fmt="%s")
    
    # Stats generationnelles : fitnesses et meilleur individu de chaque génération
    total = np.column_stack((sigma_f, minimum, OptParGen[:ngen], Ecartypes[:ngen]))
    np.savetxt(u"CMA_stats.txt", total)
    
    # Tous les individus et leur fitness
    TousLesFits = np.ravel(touslesfits)
    np.savetxt(u"CMA_population.txt", np.column_stack((ToutLeMonde, TousLesFits)))


    # Pour la courbe L
    normX = np.sum( (np.array(hof.items[0])-initial_guess)**2 ) / len(hof.items[0])
    print(normX)