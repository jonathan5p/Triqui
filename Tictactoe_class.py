import pygame
import numpy as np
import math
import gym 
from gym import spaces

class TicTacToe(gym.Env):

    def __init__(self,base_agent,render,mode,epsilon=0):
        self.rows = 4
        self.columns = 4
        self.images = []
        self.action_space = spaces.Discrete(self.rows*self.columns)
        self.observation_space = spaces.Box(low=-1.0,high=1.0,shape=(self.rows*self.columns,))
        self.base_agent=base_agent
        self.grid = self.initialize_grid()
        self.epsilon = epsilon
        self.num_envs=1
        self.mode = mode
        if mode=='primero':
            self.player = 1
            self.opponent = -1
        elif mode == 'segundo':
            self.player = -1
            self.opponent = 1
            self.opponent_step()
            
        if render==True:
            # Initializing Pygame
            pygame.init()

            # Screen
            self.WIDTH = 700
            self.gap = self.WIDTH//self.rows
            self.win = pygame.display.set_mode((self.WIDTH, self.WIDTH))
            pygame.display.set_caption("TicTacToe")

            # Colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.GRAY = (200, 200, 200)
            self.RED = (255, 0, 0)
            self.BLUE = (0, 0, 255)

                # Images
            self.X_IMAGE = pygame.transform.scale(pygame.image.load("Images/x.png"), (150, 150))
            self.O_IMAGE = pygame.transform.scale(pygame.image.load("Images/o.png"), (150, 150))

    # Fonts
            self.END_FONT = pygame.font.SysFont('courier', 40)


    def action2coord(self,action):
        row = math.floor(action/self.columns)
        col = action-row*self.columns
        return row,col 

    def opponent_step(self):
        if np.random.rand() < self.epsilon:
            opp_action = np.random.randint(self.grid.flatten().shape[0])
        else: 
            opp_action, _ = self.base_agent.predict(self.grid.flatten())
            opp_row,opp_col = self.action2coord(opp_action)
            if self.grid[opp_row,opp_col]==0:
                self.grid[opp_row,opp_col] = self.opponent


    def step(self,action):

        row,col = self.action2coord(action)
        done,reward,info=self.check_game_state()

        if done==False:

            if self.grid[row,col]==0:
                self.grid[row,col] = self.player
                done,reward,info=self.check_game_state()
            else:
                reward = -0.2
                info = {}
            
            if done!=True:
                self.opponent_step()


        observation = self.grid.flatten()
        return observation, reward, done, info

    def check_game_state(self): 
        reward = 0
        done = False
        info = {}
        if self.has_won(self.player):
            done = True
            reward = 1
        elif self.has_lost(self.player):
            done = True
            reward = -1
        elif self.has_drawn(self.player):
            done = True
            reward = 0
        
        return done,reward,info

    def reset(self):
        self.grid = self.initialize_grid()
        if self.mode == 'segundo':
            self.opponent_step()
        return self.grid.flatten()

    def render(self):
        self.win.fill(self.WHITE)
        self.draw_grid()

        # Drawing X's and O's
        for i in range(self.rows):
            y = i*self.gap+self.gap//2
            for j in range(self.columns):
                x = j*self.gap+self.gap//2
                if self.grid[i,j] == 1:
                    self.win.blit(self.X_IMAGE, (x - self.X_IMAGE.get_width() // 2, y - self.X_IMAGE.get_height() // 2))
                elif self.grid[i,j] == -1 :
                    self.win.blit(self.O_IMAGE, (x - self.O_IMAGE.get_width() // 2, y - self.O_IMAGE.get_height() // 2))
                
        pygame.display.update()

    def initialize_grid(self):
        game_array = np.zeros((self.rows,self.columns))
        return game_array

    def draw_grid(self):
        # Starting points
        x = 0
        y = 0

        for i in range(self.rows):
            x = i * self.gap

            pygame.draw.line(self.win, self.GRAY, (x, 0), (x, self.WIDTH), 3)
            pygame.draw.line(self.win, self.GRAY, (0, x), (self.WIDTH, x), 3)

    # Checking if someone has won
    def has_won(self,player):
        won = False
        row_sum = self.grid.sum(axis=0)
        col_sum = self.grid.sum(axis=1)

        if (player*self.rows in row_sum) or (player*self.columns in col_sum) or (player*self.columns == self.grid.diagonal().sum()) or (player*self.columns == np.fliplr(self.grid).diagonal().sum()): 
            won = True
        return won

    def has_lost(self,player):
        lost = False
        row_sum = self.grid.sum(axis=0)
        col_sum = self.grid.sum(axis=1)

        if (self.opponent*self.rows in row_sum) or (self.opponent*self.columns in col_sum) or (self.opponent*self.columns == self.grid.diagonal().sum()) or (self.opponent*self.columns == np.fliplr(self.grid).diagonal().sum()): 
            lost = True
        return lost

    def has_drawn(self,player): 
        draw = False
        if (self.grid==0).sum()==0:
            draw = True
        return draw

        # display_message("It's a draw!")


    def display_message(self,content):
        pygame.time.delay(500)
        self.win.fill(self.WHITE)
        end_text = self.END_FONT.render(content, 1, self.BLACK)
        self.win.blit(end_text, ((self.WIDTH - end_text.get_width()) // 2, (self.WIDTH - end_text.get_height()) // 2))
        pygame.display.update()
        pygame.time.delay(1000)
