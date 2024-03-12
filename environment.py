import random
import numpy as np
import pygame
from enum import Enum
from utils import contains, remove

# fraction of grid blocks that are obstacles
OBSTACLE_PERCENT = 0.10

# constants for screen
MAX_WIDTH, MAX_HEIGHT = 500, 500
BOTTOM_TEXT_HEIGHT = 50

# colors
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# termination status
class Status(Enum):
    ERROR = -1 # agent has gone over obstacle / out of grid
    OPEN = 0 # agent is still alive and hasn't reached the target
    SUCCESS = 1 # agent has reached the target

# moves
UP = np.array([-1, 0])
DOWN = np.array([1, 0])
LEFT = np.array([0, -1])
RIGHT = np.array([0, 1])


class Env:

    '''
    Initialize environment variables and display.
    '''
    def __init__(self, rows, columns):

        # init grid variables
        self.rows, self.columns = rows, columns
        self.all_cells = [np.array([x, y]) for x in range(self.rows) for y in range(self.columns)]

        # resize display to meet rown/columns ratio
        width = MAX_WIDTH if columns >= rows else int(columns / rows * MAX_WIDTH)
        height = MAX_HEIGHT if rows >= columns else int(rows / columns * MAX_HEIGHT)
        height += BOTTOM_TEXT_HEIGHT
        self.cell_size = width // self.columns

        # init total reward
        self.total_reward = 0

        # init pygame
        pygame.init()
        self.display = pygame.display.set_mode((width, height))

    '''
    Reset the environment to start new episode. Returns starting agent state.
    '''
    def reset(self, custom_env=False):

        if not custom_env:
            # generate random obstacles
            self.obstacles = random.sample(self.all_cells, int(OBSTACLE_PERCENT*self.rows*self.columns))

        # pick random starting position for the agent and the target
        self.agent_pos, self.target_pos = random.sample([cell for cell in self.all_cells if not contains(self.obstacles, cell)], 2)

        self.total_reward = 0

        # return starting state
        return self.get_state()
    

    '''
    Allow the user to draw a create a custom environment by choosing
    the positions of the obstacles.
    '''
    def create_custom_env(self):
        self.obstacles = []
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():

                # quit
                if event.type == pygame.QUIT:
                    running = False
                
                # left mouse click: add/remeove obstacle
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if event.pos[1] <= self.display.get_height() - BOTTOM_TEXT_HEIGHT:
                            row, col = event.pos[1] // self.cell_size, event.pos[0] // self.cell_size
                            obs = np.array([row, col])

                            if contains(self.obstacles, obs):
                                remove(self.obstacles, obs) # remove existing obstacle
                            else:
                                self.obstacles.append(obs) # add new obstacle
                
                # return: input has terminated
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        running = False

            self.render(draw_agent=False, draw_target=False)


    '''
    Place a target in a free cell.
    '''
    def place_target(self):
        free_cells = [cell for cell in self.all_cells if not contains(self.obstacles, cell) and not np.array_equal(cell, self.target_pos)]
        self.target_pos = random.choice(free_cells)


    '''
    Take an action. Returns the new state reached by taking action, 
    the reward obtained and a boolean indicating whether the game has ended.
    '''
    def step(self, action):
        # calculate new agent position
        if action == 0:
            new_agent_pos = self.agent_pos + UP
        if action == 1:
            new_agent_pos = self.agent_pos + DOWN
        if action == 2:
            new_agent_pos = self.agent_pos + LEFT
        if action == 3:
            new_agent_pos = self.agent_pos + RIGHT

        # calculate termination status
        status = self.get_termination_status(new_agent_pos)

        # calculate reward for the action
        reward = self.compute_reward(status)
        self.total_reward += reward

        # update agent position
        self.agent_pos = new_agent_pos

        # place new target if necessary
        if status == Status.SUCCESS:
            self.place_target()

        new_state = self.get_state()
        game_over = status == Status.ERROR

        return new_state, reward, game_over


    '''
    Render the environment.
    '''
    def render(self, draw_agent=True, draw_target=True):
        # background
        self.display.fill(WHITE)

        width, height = self.display.get_width(), self.display.get_height()

        # Draw grid lines
        for x in range(0, width, self.cell_size):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, width))
        for y in range(0, height, self.cell_size):
            pygame.draw.line(self.display, GRAY, (0, y), (height, y))

        # draw obstacles as black squares
        for ob in self.obstacles:
            pygame.draw.rect(self.display, BLACK, (ob[1]*self.cell_size, ob[0]*self.cell_size, self.cell_size, self.cell_size))
            
        # draw agent as blue circle
        if draw_agent:
            agent_center = self.agent_pos[1]*self.cell_size + self.cell_size//2, self.agent_pos[0]*self.cell_size + self.cell_size//2
            pygame.draw.circle(self.display, BLUE, agent_center, self.cell_size // 2)

        # draw target as yellow circle 
        if draw_target:
            target_center = self.target_pos[1]*self.cell_size + self.cell_size//2, self.target_pos[0]*self.cell_size + self.cell_size//2
            pygame.draw.circle(self.display, YELLOW, target_center, self.cell_size // 2)

        # print reward
        font = pygame.font.Font(None, 36)
        text_surface = font.render(f"Reward: {self.total_reward}", True, BLACK)
        text_rect = text_surface.get_rect()
        text_rect.bottomleft = (10, height - 10)
        self.display.blit(text_surface, text_rect)

        # update display
        pygame.display.flip()
        

    '''    
    Get current state of the agent.
    '''
    def get_state(self):

        ap, tp = self.agent_pos, self.target_pos

        # get distance from target
        dist = np.array(ap - tp)

        # can the agent move up, down, left or right?
        up = 1 if (ap[0] == 0 or contains(self.obstacles, ap + UP)) else 0
        down = 1 if (ap[0]+1 >= self.rows or contains(self.obstacles, ap + DOWN)) else 0
        left = 1 if (ap[1] == 0 or contains(self.obstacles, ap + LEFT)) else 0
        right = 1 if (ap[1]+1 >= self.columns or contains(self.obstacles, ap + RIGHT)) else 0

        available_moves = np.array([up, down, left, right])

        # the state is given by the distance from the target and the available moves
        return np.concatenate((dist, available_moves))
    

    '''
    Compute the termination status for the given position.
    '''
    def get_termination_status(self, pos):
        if pos[0] < 0 or pos[0] >= self.rows or pos[1] < 0 or pos[1] >= self.columns or contains(self.obstacles, pos):
            return Status.ERROR
        elif np.array_equal(pos, self.target_pos):
            return Status.SUCCESS
        else:
            return Status.OPEN


    '''
    Compute the reward obtained.
    '''
    def compute_reward(self, status):
        if status == Status.SUCCESS:
            return 10 # target reached
        elif status == Status.ERROR:
            return -10 # move not allowed
        elif status == Status.OPEN:
            return 0
        

    '''
    Acknowledge all the pending events to prevent pygaame window from entering "Not responding" state.
    '''
    def ignore_events(self):
        # ignore events, unless it's quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    pygame.quit()
