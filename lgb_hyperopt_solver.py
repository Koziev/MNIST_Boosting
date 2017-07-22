# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Сравнение различных реализаций градиентного бустинга для решения задачи
многоклассовой классификации изображений MNIST.
(c) Koziev Elijah inkoziev@gmail.com

Подбор гиперпараметров для LightGBM с помощью hyperopt.

Справка по XGBoost:
http://lightgbm.readthedocs.io/en/latest/python/lightgbm.html#lightgbm-package

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf

'''

import lightgbm
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

D_train = lightgbm.Dataset(X_train, y_train)
D_val = lightgbm.Dataset(X_val, y_val)
#D_test = lightgbm.Dataset(X_test, y_test)

# ---------------------------------------------------------------------

def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['application'] = 'multiclass'
    lgb_params['metric'] = 'multi_logloss'
    lgb_params['num_class'] = 10
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_leaves'] = int(space['num_leaves'])
    lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    lgb_params['max_depth'] = -1
    lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
    lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
    lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
    lgb_params['feature_fraction'] = space['feature_fraction']
    lgb_params['bagging_fraction'] = space['bagging_fraction']
    lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1

    return lgb_params

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( 'lgb-hyperopt-log.txt', 'w' )


def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nXGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str) )

    model = lightgbm.train(lgb_params,
                           D_train,
                           num_boost_round=10000,
                           # metrics='mlogloss',
                           valid_sets=D_val,
                           # valid_names='val',
                           # fobj=None,
                           # feval=None,
                           # init_model=None,
                           # feature_name='auto',
                           # categorical_feature='auto',
                           early_stopping_rounds=100,
                           # evals_result=None,
                           verbose_eval=False,
                           # learning_rates=None,
                           # keep_training_booster=False,
                           # callbacks=None
                           )

    nb_trees = model.best_iteration
    val_loss = model.best_score

    print('nb_trees={} val_loss={}'.format(nb_trees, val_loss))

    y_pred = model.predict(X_test, num_iteration=nb_trees)
    test_loss = sklearn.metrics.log_loss(y_test, y_pred, labels=list(range(10)))
    acc = sklearn.metrics.accuracy_score(y_test, numpy.argmax(y_pred, axis=1))
    print('test_loss={} test_acc={}'.format(test_loss, acc))

    log_writer.write('loss={:<7.5f} acc={} Params:{} nb_trees={}\n'.format(test_loss, acc, params_str, nb_trees ))
    log_writer.flush()

    if test_loss<cur_best_loss:
        cur_best_loss = test_loss
        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)


    return{'loss':test_loss, 'status': STATUS_OK }


# --------------------------------------------------------------------------------

space ={
        'num_leaves': hp.quniform ('num_leaves', 10, 200, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 200, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.quniform ('max_bin', 64, 512, 1),
        'bagging_freq': hp.quniform ('bagging_freq', 1, 5, 1),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),
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
