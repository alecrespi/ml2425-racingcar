import sys
import numpy as np
import tensorflow as tf

try:
    import gymnasium as gym
    from gymnasium import Env
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)


def preprocessing(obs):
    return obs / 255


def remap_action(pred):
    # action = [-1 <= steering <= 1, 0 <= gas <= 1, 0 <= braking <= 1]

    ## actions:
    # Do nothing:   0
    # Steer right:  1
    # Steer left:   2
    # Gas:          3
    # Brake:        4

    if   pred == 0:
        return np.array([0.0, 0.0, 0.0])
    elif pred == 1:
        return np.array([1.0, 0.0, 0.0])
    elif pred == 2:
        return np.array([-1.0, 0.0, 0.0])
    elif pred == 3:
        return np.array([0.0, 1.0, 0.0])
    elif pred == 4: 
        return np.array([0.0, 0.0, 1.0])


def play(env:Env, model):

    seed = 9999
    obs, _ = env.reset(seed=seed)
    
    # drop initial frames
    action0 = np.array([0, 0, 0])
    for i in range(50):
        obs,_,_,_,_ = env.step(action0)
    
    done = False
    while not done:
        obs = preprocessing(obs)

        obs_tensor = tf.convert_to_tensor(obs)
        obs_tensor = tf.expand_dims(obs_tensor, axis=0)  # Add batch dimension
        p = model.predict(obs_tensor, verbose=0)
        print("Prediction:\t", p)
        action = remap_action(np.argmax(p))
        print("Action:\t\t", action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env_arguments = {
    'domain_randomize': False,
    'continuous': True,
    'render_mode': 'human'
}

env_name = 'CarRacing-v3'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

model = tf.keras.models.load_model('models/basemodel1/best_basemodel1.h5')

play(env, model)


