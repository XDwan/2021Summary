# -*- coding: utf-8 -*-

"""
@Time : 2021/7/16
@Author : XDwan
@File : attackTrainSetGenerator
@Description : 
"""
import gc
import os

import numpy as np
from sklearn.utils import shuffle
class AttackTrainSetGenerator:
    '''
    规范数据集
    '''

    def __init__(self,shadow_group,attack_train_set_size):
        '''

        :param shadow_group:
        :param attack_train_set_size:
        '''
        self.shadow_group = shadow_group
        self.attack_train_size = attack_train_set_size
        self.X_prior,self.y_prior = shadow_group.get_prior_knowledge()

    def generate_train_set(self,target_class):
        '''

        :param target_class:
        :return:
        '''

        X_shadow,y_shadow = self.shadow_group.get_attack_data_set()
        indices = self.shadow_group.get_train_test_indices()
        posteriors,labels = [],[]

        for model_i,(train_indices_i,test_indices_i) in zip(self.shadow_group.get_models(),indices):
            X_train,X_test = self._shadow_train_test_split(
                X_shadow,y_shadow,train_indices_i,test_indices_i,target_class
            )

            if len(X_train) >0:
                posteriors.append(model_i.predict(X_train))
                labels.append(np.ones(len(X_train)))

            if len(X_test) >0:
                posteriors.append(model_i.predict(X_test))
                labels.append(
                    np.zeros(len(X_test))
                )

            if self.shadow_group.memory_efficiency:
                model_i.clear_memory()
                del model_i
                gc.collect()

        posteriors = np.concatenate(posteriors)
        labels = np.concatenate(labels)

        return shuffle(posteriors,labels)

    def _shadow_train_test_split(self,X_shadow,y_shadow,train_indices,test_indices,target_class):

        mask_train = y_shadow[train_indices] == target_class
        mask_test = y_shadow[test_indices] == target_class

        X_shadow_train = X_shadow[train_indices][mask_train]
        X_shadow_test = X_shadow[test_indices][mask_test]

        if self.X_prior is not None:
            X_shadow_train = np.concatenate(
                [self.X_prior[self.y_prior == target_class],X_shadow_train]
            )

        if type(self.attack_train_size) == dict:
            half = self.attack_train_size[target_class] // (2* self.shadow_group.size())
        else:
            half = self.attack_train_size // (2* self.shadow_group.size())

        return  X_shadow_train[:half],X_shadow_test[:half]

    def _check_train_set_size(self,X_train,X_test):
        raise NotImplementedError()