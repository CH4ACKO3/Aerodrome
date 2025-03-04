class Space3D:
    def __init__(self, dt, step_size, max_steps):
        self.initial_config = {
            'dt': dt,
            'step_size': step_size,
            'max_steps': max_steps
        }
        self.objects = []
        
    def add_object(self, obj):
        self.objects.append(obj)
        # 保存对象的初始状态
        obj.save_initial_state()
        
    def reset(self):
        # 重置环境参数
        self.__dict__.update(self.initial_config)
        
        # 重置所有对象
        for obj in self.objects:
            obj.reset() 