from __future__ import print_function

import os
import codecs
import math
import xgboost
import sklearn.metrics
import sklearn
from keras.datasets import mnist

VAL_SEED = 123456

print('Loading datasets')
(x_train0, y_train0), (x_test0, y_test0) = mnist.load_data()

print('x_train0.shape={}'.format(x_train0.shape))

X_data = x_train0.reshape( ( x_train0.shape[0], x_train0.shape[1]*x_train0.shape[1] ) )
y_data = y_train0

X_test = x_test0.reshape( ( x_test0.shape[0], x_test0.shape[1]*x_test0.shape[2] ) )

x1, x2, y1, y2 = sklearn.model_selection.train_test_split(X_data, y_data, test_size=0.2, random_state=VAL_SEED)

print('Training on {}, validating on {}'.format(x1.shape, x2.shape))

xgb_params = dict()
xgb_params['eta'] = 0.1
xgb_params['max_depth'] = 6
xgb_params['objective'] = 'multi:softprob'
xgb_params['eval_metric'] = [ 'merror', 'mlogloss']
xgb_params['num_class'] = 10
xgb_params['seed'] = VAL_SEED
xgb_params['silent'] = True

D_train = xgboost.DMatrix(x1,y1)
D_val = xgboost.DMatrix(x2,y2)
D_test = xgboost.DMatrix(X_test,y_test0)

watchlist = [(D_train, 'train'), (D_val, 'valid')]
model = xgboost.train( params=xgb_params,
                      dtrain=D_train,
                      num_boost_round=5000,
                      evals=watchlist,
                      verbose_eval=10,
                      early_stopping_rounds=50)

y_pred = model.predict(D_test, ntree_limit=model.best_ntree_limit )
test_loss = sklearn.metrics.log_loss( y_test0, y_pred, labels=list(range(10)))
acc = sklearn.metrics.accuracy_score( y_test0, y_pred )

print('test_loss={} test_acc={}'.format(test_loss, acc))



