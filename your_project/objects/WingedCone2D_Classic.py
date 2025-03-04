class WingedCone2D_Classic:
    def __init__(self, config):
        # 初始化代码...
        self.initial_state = self._capture_initial_state(config)
        
    def _capture_initial_state(self, config):
        # 保存所有初始化参数
        return {
            'pos': config['pos'].copy(),
            'vel': config['vel'].copy(),
            'ang_vel': config['ang_vel'].copy(),
            # 其他需要保存的参数...
        }
    
    def save_state(self):
        # 返回当前状态的深拷贝
        return {
            'pos': self.pos.copy(),
            'vel': self.vel.copy(),
            # 其他需要保存的状态变量...
        }
    
    def load_state(self, state):
        # 恢复状态
        self.pos = state['pos'].copy()
        self.vel = state['vel'].copy()
        # 恢复其他状态变量...
        
    def reset(self):
        # 重置到初始状态
        self.load_state(self.initial_state) 