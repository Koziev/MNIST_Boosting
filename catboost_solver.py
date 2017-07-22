# -*- coding: utf-8 -*-

from __future__ import print_function

'''
Сравнение различных реализаций градиентного бустинга для решения задачи
многоклассовой классификации изображений MNIST.
(c) Koziev Elijah inkoziev@gmail.com
'''

import catboost
import sklearn.metrics
import sklearn
import sklearn.model_selection
import numpy

import mnist_loader
import mnist_vae

#(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_loader.load_mnist()
(X_train, y_train, X_val, y_val, X_test, y_test ) =  mnist_vae.load_mnist()

D_train = catboost.Pool(X_train, y_train)
D_val = catboost.Pool(X_val, y_val)

params = dict()
params['learning_rate'] = 0.10
params['depth'] = 6
params['l2_leaf_reg'] = 4
params['rsm'] = 1.0

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

print('nb_trees={}'.format(model.get_tree_count()))


y_pred = model.predict_proba(X_test)

test_loss = sklearn.metrics.log_loss( y_test, y_pred, labels=list(range(10)))

acc = sklearn.metrics.accuracy_score( y_test, numpy.argmax( y_pred, axis=1 ) )

print('test_loss={} test_acc={}'.format(test_loss, acc))


