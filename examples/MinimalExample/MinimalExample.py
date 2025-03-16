import aerodrome

if __name__ == "__main__":
    env = aerodrome.make("minimal-v0") # 创建 Python 环境
    result = env.reset() # 重置环境
    print(result) # 打印重置后返回的结果

    while True:
        try:
            action = input("Enter an action value: ") # 从用户输入获取动作（一个整数）
        except KeyboardInterrupt:
            env.close()
            break
        
        try:
            action = int(action) # 将用户输入的动作转换为整数
        except ValueError:
            print("Invalid action value")
            continue
        result = env.step(action) # 调用环境 step 方法，并接收返回结果
        print(result) # 打印返回结果

        if result["py_state"] > 10: # 如果 Python 环境内部状态大于 10（大于 10 步），则重置环境
            result = env.reset()
            print(result)