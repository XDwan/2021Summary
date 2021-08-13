import os
import gc
import numpy as np
class Attacker:
    '''
    该类包括针对几个目标类的一系列攻击模型
    在攻击一个类之前，attacker必须被训练，对该类的攻击才能执行
    '''

    def __init__(self,attack_model_factory,target_classes,memory_efficiency=False):
        '''
        创建一个攻击者对象，其中包含几个攻击模型
        每一个攻击模型由一个模型工厂创建
        :param attack_model_factory: 生产攻击模型的模型工厂
        :param target_classes:
        :param memory_efficiency:
        '''

        self.__model_factory = attack_model_factory
        self.memory_efficiency = memory_efficiency
        self.attack_models = {}
        self.path_template = os.path.join("/", "tmp", "mia_attack_model_{}.h5")
        self.target_classes = target_classes

    def train_attack_model(
            self, target_class, X_train, y_train, X_val=None, y_val=None
    ):
        """
        训练攻击者对特定目标类进行成员关系推理攻击
        :param X_train: 该类的训练集
        :param y_train: 0或1的标签
        :param target_class: 目标模型中要攻击的类
        """
        model = self.__model_factory.create()
        single_attack_history = model.fit(X_train, y_train, X_val, y_val)
        if self.memory_efficiency:
            model.save(self.path_template.format(target_class))
        else:
            self.attack_models[target_class] = model

        return single_attack_history

    def attack_target_class(self,X_test,target_class):
        '''
        攻击输出目标模型关于一个特定目标类的X_train
        :param X_test: 目标模型的输出
        :param target_class: 攻击的类
        :return: 成员关系预测
        '''

        model = self.get_model(target_class)
        predictions = model.predict(X_test)

        if self.memory_efficiency:
            model.clear_memory()
            del model
            gc.collect()

        return predictions

    def evaluate_attack(self,X_test,y_test,target_class,metrics):
        '''

        :param X_test:
        :param y_test:
        :param target_class:
        :param metrics:
        :return:
        '''

        y_pred = self.attack_target_class(X_test,target_class)
        y_hot_pred = np.argmax(y_pred,axis=1)
        results = [metric(y_test,y_hot_pred) for metric in metrics]
        return  results, y_pred

    def get_models(self):
        '''

        :return:
        '''

        for target_class in self.target_classes:
            model = self.get_model(target_class)
            yield model
            del model
            gc.collect()

    def get_model(self,target_class):
        '''

        :param target_class:
        :return:
        '''

        if self.memory_efficiency:
            model = self.__model_factory.create()
            model.load(self.path_template.format(target_class))
        else:
            model = self.attack_models[target_class]
        return model

