

class AttackTrainSetGenerator:
    '''
    用于简化训练集生成器的类，更好的适应基于影子训练的MI攻击模型
    '''

    def __init__(self,shadow_group,attack_train_set_size):
        '''
        构造生成器
        :param shadow_group: 一组用于生成的训练好的影子模型
        :param attack_train_set_size: 训练集结果的大小 或者每一个攻击模型独立的特定训练大小
        生成器总是从训练集和测试集中抽取相同数量的样本来保持平衡
        因此 如果attack_train_set_size不是 2*shadow_group.size()，这个返回的数据集可能有更少的样本
        '''

        self.shadow_group = shadow_group
        self.attack_train_size = attack_train_set_size
        self.X_prior, self.y_prior = shadow_group.get_prior_knowlegdge()

    def generate_train_set(self,target_class):
        '''
        对于目标类生成一组训练集
        :param target_class: 要创建攻击集的类
        :return: 数据和标签的元组
        '''

        X_shadow,y_shadow = self.shadow_group.get_attack_data_set()
