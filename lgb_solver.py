# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Сравнение различных реализаций градиентного бустинга для решения задачи
многоклассовой классификации изображений MNIST.
(c) Koziev Elijah inkoziev@gmail.com
'''

import os
import codecs
import math
import lightgbm
import sklearn.metrics
import sklearn
import numpy
import mnist_loader
import mnist_vae

#(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_loader.load_mnist()
(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_vae.load_mnist()

D_train = lightgbm.Dataset(X_train, y_train)
D_val = lightgbm.Dataset(X_val, y_val)


lgb_params = dict()
lgb_params['boosting_type'] = 'gbdt'  # space['boosting_type'], # 'gbdt', # gbdt | dart | goss
# px['objective'] ='multi:softprob'
lgb_params['application'] = 'multiclass'
lgb_params['metric'] = 'multi_logloss'
lgb_params['num_class'] = 10
lgb_params['learning_rate'] = 0.05
lgb_params['num_leaves'] = 100
lgb_params['min_data_in_leaf'] = 100
lgb_params['min_sum_hessian_in_leaf'] = 1e-3
lgb_params['max_depth'] = -1
lgb_params['lambda_l1'] = 0.0  # space['lambda_l1'],
lgb_params['lambda_l2'] = 0.0  # space['lambda_l2'],
lgb_params['max_bin'] = 256
lgb_params['feature_fraction'] = 0.7
lgb_params['bagging_fraction'] = 0.7
lgb_params['bagging_freq'] = 1


model = lightgbm.train(lgb_params,
                       D_train,
                       num_boost_round=5000,
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

print('nb_trees={}'.format(nb_trees))

y_pred = model.predict(X_test, num_iteration=nb_trees )
test_loss = sklearn.metrics.log_loss( y_test, y_pred, labels=list(range(10)))

acc = sklearn.metrics.accuracy_score( y_test, numpy.argmax( y_pred, axis=1 ) )

print('test_loss={} test_acc={}'.format(test_loss, acc))


