import flappy_bird_gymnasium
import gymnasium
#env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False) #initially we will code on simpler env to be able debug our code
env = gymnasium.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample() # sample will give back 0 and 1 that is action of bird 0 means nothing 1 mean action

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    # obs is next state info, reward from last round
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()


# now we will train a deep qn network it will have 12 input dim as returned by obs varibale that is hieght of last pole, and etc 
# we are only using a single hidden layer
# output layer is expected action that 0 or 1 so two layer
# this network is refered to as policy network