import aerodrome
import aerodrome.registration
# print(aerodrome.registry)
# aerodrome.registration.register("example-v0", "aerodrome.envs.example:ExampleEnv")

env = aerodrome.make("example-v0")
env.step(1)
env.step(2)
env.step(2)
env.step(2)
env.step(2)
env.reset()
env.step(2)
