# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Сравнение различных реализаций градиентного бустинга для решения задачи
многоклассовой классификации изображений MNIST.
(c) Koziev Elijah inkoziev@gmail.com


Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf

'''

import sklearn
import numpy
import catboost
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import colorama
import numpy as np
import mnist_loader
import mnist_vae


# кол-во случайных наборов гиперпараметров
N_HYPEROPT_PROBES = 500

# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest

# ----------------------------------------------------------

colorama.init()


#(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_loader.load_mnist()
(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_vae.load_mnist()

D_train = catboost.Pool(X_train, y_train)
D_val = catboost.Pool(X_val, y_val)

# ---------------------------------------------------------------------

def get_catboost_params(space):
    params = dict()
    params['learning_rate'] = space['learning_rate']
    params['depth'] = int(space['depth'])
    params['l2_leaf_reg'] = space['l2_leaf_reg']
    params['rsm'] = space['rsm']
    return params

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( 'catboost-hyperopt-log.txt', 'w' )


def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    params = get_catboost_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str) )

    model = catboost.CatBoostClassifier(iterations=5000,
                                        learning_rate=params['learning_rate'],
                                        depth=int(params['depth']),
                                        loss_function='MultiClass',
                                        use_best_model=True,
                                        eval_metric='MultiClass',
                                        l2_leaf_reg=params['l2_leaf_reg'],
                                        auto_stop_pval=1e-3,
                                        random_seed=123456,
                                        verbose=False
                                        )
    model.fit(D_train, eval_set=D_val, verbose=True)
    nb_trees = model.get_tree_count()

    print('nb_trees={}'.format(nb_trees))

    y_pred = model.predict_proba(X_test)

    test_loss = sklearn.metrics.log_loss(y_test, y_pred, labels=list(range(10)))
    acc = sklearn.metrics.accuracy_score(y_test, numpy.argmax(y_pred, axis=1))

    log_writer.write('loss={:<7.5f} acc={} Params:{} nb_trees={}\n'.format(test_loss, acc, params_str, nb_trees ))
    log_writer.flush()

    if test_loss<cur_best_loss:
        cur_best_loss = test_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)


    return{'loss':test_loss, 'status': STATUS_OK }


# --------------------------------------------------------------------------------

space ={
        'depth': hp.quniform("depth", 4, 7, 1),
        'rsm': hp.uniform ('rsm', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -3.0, -0.7),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
       }


trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')
