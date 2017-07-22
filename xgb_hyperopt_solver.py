# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Сравнение различных реализаций градиентного бустинга для решения задачи
многоклассовой классификации изображений MNIST.
(c) Koziev Elijah inkoziev@gmail.com

Подбор гиперпараметров для XGBoost с помощью hyperopt.

Справка по XGBoost:
http://xgboost.readthedocs.io/en/latest/

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf

'''

import xgboost
import sklearn
import numpy
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

D_train = xgboost.DMatrix(X_train, y_train)
D_val = xgboost.DMatrix(X_val, y_val)
D_test = xgboost.DMatrix(X_test, y_test)
watchlist = [(D_train, 'train'), (D_val, 'valid')]

# ---------------------------------------------------------------------

def get_xgboost_params(space):
    _max_depth = int(space['max_depth'])
    _min_child_weight = space['min_child_weight']
    _subsample = space['subsample']
    _gamma = space['gamma'] if 'gamma' in space else 0.01
    _eta = space['eta']
    _seed = space['seed'] if 'seed' in space else 123456
    _colsample_bytree = space['colsample_bytree']
    _colsample_bylevel = space['colsample_bylevel']
    booster = space['booster'] if 'booster' in space else 'gbtree'

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])

    xgb_params = {
        'booster': booster,
        'subsample': _subsample,
        'max_depth': _max_depth,
        'seed': _seed,
        'min_child_weight': _min_child_weight,
        'eta': _eta,
        'gamma': _gamma,
        'colsample_bytree': _colsample_bytree,
        'colsample_bylevel': _colsample_bylevel,
        'scale_pos_weight': 1.0,
        'eval_metric': 'logloss', #'auc',  # 'logloss',
        'objective': 'binary:logistic',
        'silent': 1,
    }

    xgb_params['updater'] = 'grow_gpu'

    xgb_params['objective'] = 'multi:softprob'
    xgb_params['eval_metric'] = ['merror', 'mlogloss']
    xgb_params['num_class'] = 10

    return xgb_params

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( 'xgb-hyperopt-log.txt', 'w' )


def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nXGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    xgb_params = get_xgboost_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str) )

    model = xgboost.train(params=xgb_params,
                          dtrain=D_train,
                          num_boost_round=5000,
                          evals=watchlist,
                          verbose_eval=False,
                          early_stopping_rounds=50)

    print('nb_trees={} val_loss={:7.5f}'.format(model.best_ntree_limit, model.best_score))
    #loss = model.best_score
    nb_trees = model.best_ntree_limit
    y_pred = model.predict(D_test, ntree_limit=nb_trees)
    test_loss = sklearn.metrics.log_loss(y_test, y_pred, labels=list(range(10)))
    acc = sklearn.metrics.accuracy_score(y_test, np.argmax(y_pred, axis=1))
    print('test_loss={} test_acc={}'.format(test_loss, acc))

    log_writer.write('loss={:<7.5f} Params:{} nb_trees={}\n'.format(test_loss, params_str, nb_trees ))
    log_writer.flush()

    if test_loss<cur_best_loss:
        cur_best_loss = test_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)


    return{'loss':test_loss, 'status': STATUS_OK }


# --------------------------------------------------------------------------------

space ={
        #'booster': hp.choice( 'booster',  ['dart', 'gbtree'] ),
        'max_depth': hp.quniform("max_depth", 4, 7, 1),
        'min_child_weight': hp.quniform ('min_child_weight', 1, 20, 1),
        'subsample': hp.uniform ('subsample', 0.75, 1.0),
        #'gamma': hp.uniform('gamma', 0.0, 0.5),
        'gamma': hp.loguniform('gamma', -5.0, 0.0),
        #'eta': hp.uniform('eta', 0.005, 0.018),
        'eta': hp.loguniform('eta', -4.6, -2.3),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.70, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.70, 1.0),
        #'seed': hp.randint('seed', 2000000)
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
