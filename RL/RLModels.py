import pandas as pd
import numpy as np
import os
import time
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

class ModelsFactory:
    def train_A2C(self, env_train, model_name, timesteps=25000, TRAINED_MODEL_DIR = 'models'):
        """A2C model"""

        #Trae el tiempo actual
        start = time.time()
        
        #Instancia el modelo, ingresa entorno
        model = A2C('MlpPolicy', env_train, verbose=0)
        #Entrenamiento
        model.learn(total_timesteps=timesteps)
        #Trae el tiempo actual
        end = time.time()

        #Guarda modelo
        model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
        #Estado
        print('Training time (A2C): ', (end - start) / 60, ' minutes')
        return model

    def train_load_A2C(self, env_train, model_name, TRAINED_MODEL_DIR = 'models'):
        #Guarda modelo
        model = A2C.load(f"{TRAINED_MODEL_DIR}/{model_name}", env=env_train)
        return model 

    def train_DDPG(self, env_train, model_name, timesteps=10000, TRAINED_MODEL_DIR = 'models'):
        """DDPG model"""

        # add the noise objects for DDPG
        #Tamaño de espacio de accion
        n_actions = env_train.action_space.shape[-1]
        param_noise = None
        
        #Ruido de accion
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        start = time.time()
        #param_noise=param_noise,
        model = DDPG('MlpPolicy', env_train, action_noise=action_noise)
        model.learn(total_timesteps=timesteps)
        end = time.time()

        model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (DDPG): ', (end-start)/60,' minutes')
        return model

    def train_load_DDPG(self, env_train, model_name, TRAINED_MODEL_DIR = 'models'):
        #Guarda modelo
        model = DDPG.load(f"{TRAINED_MODEL_DIR}/{model_name}", env=env_train)
        return model 

    def train_PPO(self, env_train, model_name, timesteps=50000, TRAINED_MODEL_DIR = 'models'):
        """PPO model"""

        start = time.time()
        model = PPO('MlpPolicy', env_train, ent_coef = 0.005, batch_size = 8)
        #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

        model.learn(total_timesteps=timesteps)
        end = time.time()

        model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (PPO): ', (end - start) / 60, ' minutes')
        return model

    def train_load_PPO(self, env_train, model_name, TRAINED_MODEL_DIR = 'models'):
        #Guarda modelo
        model = PPO.load(f"{TRAINED_MODEL_DIR}/{model_name}", env=env_train)
        return model 