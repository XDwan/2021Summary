# -*- coding: utf-8 -*-

"""
@Time : 2021/7/16
@Author : XDwan
@File : attackTestSetGenerator
@Description : 
"""

import numpy as np

class AttackTestSetGenerator:
    '''

    '''

    def __init__(self,target_model,test_set_size):
        '''

        :param target_model:
        :param test_set_size:
        '''
        self.target_model = target_model
        self.test_set_size = test_set_size

    def generate_test_set(self,target_class,X_target,y_target,X_test,y_test):
        '''

        :param target_class:
        :param X_target:
        :param y_target:
        :param X_test:
        :param y_test:
        :return:
        '''

        mask1 = np.array(y_target) == target_class
        mask2 = np.array(y_test) == target_class
        index_map1 = np.where(mask1)[0]
        index_map2 = np.where(mask2)[0]

        X_target_for_class = np.array(X_target)[mask1]
        X_test_for_class = np.array(X_test)[mask2]
        self._check_test_set_size(X_target, X_test, target_class)

        if type(self.test_set_size) is dict:
            test_set_size = self.test_set_size[target_class]
        else:
            test_set_size = self.test_set_size

        half_N = test_set_size // 2
        indices = np.random.choice(len(X_target_for_class), half_N, replace=False)
        X_target_for_class = X_target_for_class[indices]
        in_indices = index_map1[indices]

        indices = np.random.choice(len(X_test_for_class), half_N, replace=False)
        X_test_for_class = X_test_for_class[indices]
        out_indices = index_map2[indices]

        X_attack_test = np.concatenate(
            [
                self.target_model.predict(X_target_for_class),
                self.target_model.predict(X_test_for_class),
            ]
        )
        y_attack_test = np.concatenate([np.ones(half_N), np.zeros(half_N)])

        return X_attack_test, y_attack_test, (in_indices, out_indices)

    def _check_test_set_size(self, X_train, X_test, target_class):
        '''

        :param X_train:
        :param X_test:
        :param target_class:
        :return:
        '''

        if type(self.test_set_size) is dict:
            test_set_size = self.test_set_size[target_class]
        else:
            test_set_size = self.test_set_size

        if test_set_size // 2 > len(X_train) or test_set_size // 2 > len(X_test):
            raise ValueError(
                f"Cannot create a balanced test set of size {self.test_set_size} using"
                f" X_train of length {len(X_train)} and X_test of length {len(X_test)}"
                f" for class {target_class}."
            )
