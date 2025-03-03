import numpy as np

class PerceptionPosition:
    def __init__(self, random_range):
        '''
        根据随机范围，随机返回位置的值
        :param random_range: 随机范围, shape = (n,)
        '''
        self.random_range = random_range if type(random_range) == np.ndarray else np.array(random_range)
        self.random_range = np.array([self.random_range, self.random_range, np.zeros_like(self.random_range)]).T
        self.q = np.zeros((random_range.shape[0], 3))  
        self.random_num = np.zeros((random_range.shape[0], random_range.shape[0], 3))

        # 记录历史数据
        self._history = []
    
    def updata_actrual_q(self, q):
        '''
        设置实际的位置
        :param q: 实际的位置, shape = (n, 3)
        '''
        self.q = q if type(q) == np.ndarray else np.array(q)
        # for observer in range(self.random_range.shape[0]):
        #     random_x = np.random.uniform(-self.random_range[observer][0], self.random_range[observer][0], self.q.shape[0])
        #     random_y = np.random.uniform(-self.random_range[observer][1], self.random_range[observer][1], self.q.shape[0])
        #     self.random_num[observer] = np.array([random_x, random_y]).T
        
        # 向量化生成随机数
        self.random_num = np.random.uniform(
            -self.random_range[:, np.newaxis, :],
            self.random_range[:, np.newaxis, :],
            (self.random_range.shape[0], self.q.shape[0], 3)
        )

        # 记录历史数据
        self._history.append(self.q.copy() + self.random_num.copy())
        

    def get_observation(self, observer: int):
        '''
        获取观测值
        :param observer: 观察者的编号
        :return: 观测值
        '''
        return self._history[-1][observer]
    
    def __call__(self, observer: int):
        return self.get_observation(observer)
    
    @property
    def history(self) -> list:
        '''
        观测历史数据, shape = (round, n, n, 3)
        '''
        return self._history