import aerodrome

if __name__ == "__main__":
    env = aerodrome.make("minimal-v0")
    result = env.reset()
    print(result)

    while True:
        try:
            action = input("Enter an action value: ")
        except KeyboardInterrupt:
            env.close()
            break
        
        try:
            action = int(action)
        except ValueError:
            print("Invalid action value")
            continue
        result = env.step(action)
        print(result)

        if result["py_state"] > 10:
            result = env.reset()
            print(result)