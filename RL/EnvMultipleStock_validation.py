import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


class StockEnvValidation(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                    df, 
                    day = 0, 
                    turbulence_threshold=140, 
                    iteration='',
                    HMAX_NORMALIZE=100,
                    INITIAL_ACCOUNT_BALANCE=1000000,
                    STOCK_DIM=10,
                    TRANSACTION_FEE_PERCENT=0.001,
                    PATH_RESULTS = '/content/results',
                    logsActive = False):

        self.logsActive = logsActive
        self.PATH_RESULTS = PATH_RESULTS
        self.HMAX_NORMALIZE = HMAX_NORMALIZE
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.STOCK_DIM = STOCK_DIM
        self.TRANSACTION_FEE_PERCENT = TRANSACTION_FEE_PERCENT

        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.STOCK_DIM*6+1,))
        # load data from a pandas dataframe
        self.lf = list(self.df.index.unique())
        self.lf.sort()
        self.day_str = self.lf[self.day] 
        self.data = self.df.loc[self.day_str,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.state = [self.INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0]*self.STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [self.INITIAL_ACCOUNT_BALANCE]
        self.date_memory = [self.data.datadate.values[0]]
        self.rewards_memory = []
        #self.reset()
        self._seed()
        
        self.iteration=iteration


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            if self.state[index+self.STOCK_DIM+1] > 0:
                #update balance
                minAction = min(abs(action),self.state[index+self.STOCK_DIM+1])

                self.state[0] += self.state[index+1] * minAction * (1- self.TRANSACTION_FEE_PERCENT)
                
                self.state[index+self.STOCK_DIM+1] -= minAction

                self.cost += self.state[index+1] * minAction * self.TRANSACTION_FEE_PERCENT
                
                self.trades+=1
            else:
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions 
            if self.state[index+self.STOCK_DIM+1] > 0:
                #update balance

                self.state[0] += self.state[index+1] * self.state[index+self.STOCK_DIM+1] * (1- self.TRANSACTION_FEE_PERCENT)
                
                self.state[index+self.STOCK_DIM+1] = 0
                
                self.cost += self.state[index+1] * self.state[index+self.STOCK_DIM+1] * self.TRANSACTION_FEE_PERCENT
                
                self.trades+=1
            else:
                pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index+1]

            # print('available_amount:{}'.format(available_amount))
            #update balance
            minAviableAmount = min(available_amount, action)

            self.state[0] -= self.state[index+1] * minAviableAmount * (1+ self.TRANSACTION_FEE_PERCENT)

            self.state[index+self.STOCK_DIM+1] += minAviableAmount
            
            self.cost+=self.state[index+1] * minAviableAmount * self.TRANSACTION_FEE_PERCENT

            self.trades+=1
        else:
            # if turbulence goes over threshold, just stop buying
            pass

        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            #plt.plot(self.asset_memory,'r')
            #plt.savefig(self.PATH_RESULTS+'/account_value_validation_{}.png'.format(self.iteration))
            #plt.close()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(self.PATH_RESULTS+'/account_value_validation_{}.csv'.format(self.iteration))
            
            df_dates = pd.DataFrame(self.date_memory)
            df_dates.to_csv(self.PATH_RESULTS+'/account_date_validation_{}.csv'.format(self.iteration))
            
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(self.STOCK_DIM+1)]) * np.array(self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)]))
            
            #print("previous_total_asset:{}".format(self.asset_memory[0]))           

            #print("end_total_asset:{}".format(end_total_asset))
            #print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(self.STOCK_DIM+1)])*np.array(self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)]))- self.asset_memory[0] ))
            #print("total_cost: ", self.cost)
            #print("total trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (4**0.5)*df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            print("Sharpe: ",sharpe)
            
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(self.PATH_RESULTS+'/account_rewards_validation_{}.csv'.format(self.iteration))
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            with open(self.PATH_RESULTS+'/obs_validation_{}.pkl'.format(self.iteration), 'wb') as f:  
                pickle.dump(self.state, f)
    
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * self.HMAX_NORMALIZE
            #actions = (actions.astype(int))
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-self.HMAX_NORMALIZE]*self.STOCK_DIM)
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(self.STOCK_DIM+1)]) * np.array(self.state[(self.STOCK_DIM+1) : (self.STOCK_DIM*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.day_str = self.lf[self.day] 
            self.data = self.df.loc[self.day_str,:]         
            self.turbulence = self.data['turbulence'].values[0]
            #print(self.turbulence)
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(self.STOCK_DIM+1)]) * np.array(self.state[(self.STOCK_DIM+1) : (self.STOCK_DIM*2+1)]))
            
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.data.datadate.values[0])


        return self.state, self.reward, self.terminal, {}


    def reset(self):  
        self.asset_memory = [self.INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.day_str = self.lf[self.day] 
        self.data = self.df.loc[self.day_str,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        #self.iteration=self.iteration
        self.date_memory = [self.data.datadate.values[0]]    
        self.rewards_memory = []

        self.log([self.INITIAL_ACCOUNT_BALANCE])
        self.log(self.data.close.values.tolist())
        self.log([0]*self.STOCK_DIM)
        self.log(self.data.macd.values.tolist())
        self.log(self.data.rsi.values.tolist())
        self.log(self.data.cci.values.tolist())
        self.log(self.data.adx.values.tolist() )


        #initiate state
        self.state = [self.INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0]*self.STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist()  + \
                      self.data.cci.values.tolist()  + \
                      self.data.adx.values.tolist() 
        self.log(self.state)
        return self.state
    

    def render(self, mode='human',close=False):
        return self.state
        
    def log(self, s):
        if(self.logsActive):
            print(s)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]