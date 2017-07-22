# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Сравнение различных реализаций градиентного бустинга для решения задачи
многоклассовой классификации изображений MNIST.
(c) Koziev Elijah inkoziev@gmail.com
'''

import xgboost
import sklearn
import numpy
import mnist_loader
import mnist_vae

#(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_loader.load_mnist()
(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_vae.load_mnist()

D_train = xgboost.DMatrix(X_train, y_train)
D_val = xgboost.DMatrix(X_val, y_val)
D_test = xgboost.DMatrix(X_test, y_test)

xgb_params = dict()
xgb_params['eta'] = 0.0365253500815
xgb_params['max_depth'] = 5
xgb_params['subsample'] = 0.789639281187
xgb_params['min_child_weight'] = 3
xgb_params['gamma'] = 0.271712091643
xgb_params['colsample_bytree'] = 0.774391443402
xgb_params['colsample_bylevel'] = 0.79407015729
xgb_params['objective'] = 'multi:softprob'
xgb_params['eval_metric'] = [ 'merror', 'mlogloss']
xgb_params['num_class'] = 10
xgb_params['seed'] = 123456
xgb_params['silent'] = True
xgb_params['updater'] = 'grow_gpu'

watchlist = [(D_train, 'train'), (D_val, 'valid')]

model = xgboost.train( params=xgb_params,
                      dtrain=D_train,
                      num_boost_round=5000,
                      evals=watchlist,
                      verbose_eval=10,
                      early_stopping_rounds=50)

print('nb_trees={}'.format(model.best_ntree_limit))

y_pred = model.predict(D_test, ntree_limit=model.best_ntree_limit )
test_loss = sklearn.metrics.log_loss( y_test, y_pred, labels=list(range(10)))
acc = sklearn.metrics.accuracy_score( y_test, numpy.argmax( y_pred, axis=1 ) )

print('test_loss={} test_acc={}'.format(test_loss, acc))


