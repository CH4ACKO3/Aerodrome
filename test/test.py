import aerodrome
import aerodrome.registration
# print(aerodrome.registry)
# aerodrome.registration.register("example-v0", "aerodrome.envs.example:ExampleEnv")

env = aerodrome.make("cartpole-v0")
obs, info = env.reset()
print(obs, info)
for i in range(100):
    obs, reward, terminated, truncated, info = env.step(1)
    if terminated or truncated:
        break
    print(obs, reward, terminated, truncated, info)
