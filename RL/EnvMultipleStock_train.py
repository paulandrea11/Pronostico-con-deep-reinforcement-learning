import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                    df,
                    day = 0,
                    HMAX_NORMALIZE=100,
                    INITIAL_ACCOUNT_BALANCE=1000000,
                    STOCK_DIM=10,
                    TRANSACTION_FEE_PERCENT=0.001,
                    logsActive = False,
                    PATH_RESULTS = '/content/results',
                    logCounterInterval = 1):

        self.logsActive = logsActive
        self.PATH_RESULTS = PATH_RESULTS
        self.logCounter = 0
        self.logCounterInterval = logCounterInterval
        self.HMAX_NORMALIZE = HMAX_NORMALIZE
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.STOCK_DIM = STOCK_DIM
        self.TRANSACTION_FEE_PERCENT = TRANSACTION_FEE_PERCENT

        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df

        # action_space normalization and shape is self.STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.STOCK_DIM*6+1,))
        # load data from a pandas dataframe
        self.lf = list(self.df.index.unique())
        self.lf.sort()
        self.lfLen = len(self.lf)
        print('** Len data: ',self.lfLen)
        self.day_str = self.lf[self.day] 
        self.data = self.df.loc[self.day_str,:]
        self.terminal = False             
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
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [self.INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        #self.reset()
        self._seed()


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index+self.STOCK_DIM+1] > 0:
            #update balance
            minAction = min( abs(action), self.state[index + self.STOCK_DIM + 1] )

            self.state[ 0 ] += self.state[ index + 1 ] * minAction * (1 - self.TRANSACTION_FEE_PERCENT)

            self.state[ index + self.STOCK_DIM + 1 ] -= minAction

            self.cost += self.state[ index + 1 ] * minAction * self.TRANSACTION_FEE_PERCENT
            
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[ 0 ] // self.state[ index + 1 ]

        # self.log('available_amount:{}'.format(available_amount))
        #update balance
        minAviableAmount = min(available_amount, action)

        self.state[0] -= self.state[index+1] * minAviableAmount * ( 1 + self.TRANSACTION_FEE_PERCENT)

        self.state[ index + self.STOCK_DIM + 1 ] += minAviableAmount

        self.cost += self.state[index+1] * minAviableAmount * self.TRANSACTION_FEE_PERCENT

        self.trades += 1

        
    def step(self, actions):
        self.logCounter = ((self.logCounter + 1) % self.logCounterInterval)
        if(self.logsActive):
            self.logsActive = self.logCounter == 0

        self.terminal = self.day >= self.lfLen-1
        if self.terminal:
            print("** Terminal ================================= 1")
            #print(self.PATH_RESULTS+'/account_value_train.png')
            #try:
            #    plt.plot(self.asset_memory,'r')
            #    plt.savefig(self.PATH_RESULTS+'/account_value_train.png')
            #    plt.close()
            #except Exception as err:
            #    print(f"Unexpected {err=}, {type(err)=}")
            #    raise
            #print("Save png")

            end_total_asset = self.state[0] + sum(np.array(self.state[1:(self.STOCK_DIM+1)]) * np.array(self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)]))
            
            self.log("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(self.PATH_RESULTS+'/account_value_train.csv')

            self.log("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(self.STOCK_DIM+1)])*np.array(self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)]))- self.INITIAL_ACCOUNT_BALANCE ))
            self.log("total_cost: {}".format(self.cost))
            self.log("total_trades: {}".format(self.trades))
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            
            self.log(df_total_value['daily_return'])
            
            sharpe = (252**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            
            self.log("Sharpe trained terminal: {}".format(sharpe))
            self.log("=================================")

            df_rewards = pd.DataFrame(self.rewards_memory)
            
            df_rewards.to_csv(self.PATH_RESULTS+'/account_rewards_train.csv')
            
            #self.log('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)

            return self.state, self.reward, self.terminal,{}

        else:
            #self.log(np.array(self.state[1:11]))
            #self.log("--------------")

            self.log("** Step =================================")
            if(self.day % 10000 == 0):
                print(f"Step: {self.day}")
            self.log(f"Actions: {actions}")
            actions = actions * self.HMAX_NORMALIZE
            self.log(f"Actions: {actions}")
            
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:(self.STOCK_DIM+1)]) * np.array(self.state[(self.STOCK_DIM+1) : (self.STOCK_DIM*2+1)]))
            self.log("begin_total_asset: {begin_total_asset}")
            
            argsort_actions = np.argsort(actions)
            self.log("argsort_actions: {argsort_actions}")

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            self.log("sell_index: {sell_index}")
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            self.log("buy_index: {buy_index}")

            for index in sell_index:
                self.log(f"take sell action: {actions[index]}")
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self.log(f"take buy action: {actions[index]}")
                self._buy_stock(index, actions[index])

            self.day += 1
            self.day_str = self.lf[self.day] 
            self.data = self.df.loc[self.day_str,:]         
            self.log(f"day_str: {self.day_str}")
            
            #load next state
            self.log(f"stock_shares: {self.state[0]}")

            self.state =  [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            self.log(f"state: {self.state}")
            
            end_total_asset = self.state[0] + sum(np.array(self.state[1:(self.STOCK_DIM+1)]) * np.array(self.state[(self.STOCK_DIM+1):(self.STOCK_DIM*2+1)]))
            
            self.log(f"Step end_total_asset:{end_total_asset}")
            
            self.reward = end_total_asset - begin_total_asset            
            self.log(f"Step reward: {self.reward}")
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)

            self.log("=================================")
        return self.state, self.reward, self.terminal, {}


    def reset(self):         
        self.log(f"reset")
        self.asset_memory = [self.INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.day_str = self.lf[self.day]
        self.data = self.df.loc[self.day_str,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state

        self.state = [self.INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0]*self.STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() 
        self.log(self.state)
        # iteration += 1 
        return self.state
    

    def render(self, mode='human'):
        return self.state
    
    def log(self, s):
        if(self.logsActive):
            print(s)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]