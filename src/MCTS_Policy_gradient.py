import time
import Game_logic
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import Brain
import math
import time
def max_in_board(board_list) :
    """Find the maximum value in the game board.
    
    Args:
        board_list: 2D list representing the game board
        
    Returns:
        Maximum value found
    """
    return max(max(row) for row in board_list)
def monotomy (board):
    distance = 0
    consecutives = True
    for k in range(len(board)**2-1):
        i = k//len(board)
        j = k%len(board)
        j = j*(i%2) + (len(board)-1-j)*(1 - i%2)
        l = (k+1)//len(board)
        m = (k+1)%len(board)
        m = m*(l%2) + (len(board)-1-m)*(1 - l%2)
        if ( board[i][j] > board[l][m]) and consecutives:
            distance += 1
        else:
            consecutives = False
            break
    return  distance
def smoothness (board):
    smoothness  = 0
    for i in range(len(board)-1):
        for j in range(len(board)-1):
            if board[i][j] != 0 and board[i][j+1] != 0:
                smoothness += abs(math.log2(board[i][j]) - math.log2(board[i][j+1]) )**2
            if board[i][j] != 0 and board[i+1][j] != 0:
                smoothness += abs(math.log2(board[i][j]) - math.log2(board[i+1][j]) )**2
    return  smoothness 
def evaluate_board(board,a=10,b=1,c=1):
    zero_count = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                zero_count += 1
    score = (
    math.log2(max_in_board(board)) *(a*(board[0][0] == max_in_board(board)) )+
    b*zero_count +
    c*monotomy(board)
    #- 0.3*smoothness(board)
    )
    return score
# Neural Network for predicting move probabilities
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=16, output_size=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return torch.softmax(self.fc4(x), dim=-1)
    def save(self, filename):
        torch.save(self.state_dict(), filename)
    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def optimize(self,board_input,reward,chosen_move=None,alpha=0.01,gamma=0.9):
        output = self.forward(board_input)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        if chosen_move is not None:
            chosen_move_index = Brain.convert_move_to_index(chosen_move)
            selected_component = output[chosen_move_index]
            selected_component.backward()
            with torch.no_grad():
                for param in self.parameters():
                    param +=  alpha *gamma*reward* param.grad/selected_component
                    param.grad.zero_()  # Reset gradients after update
    def learn(self ,learning_rate=0.001,gamma=0.9,evaluation_function=evaluate_board, visualize=False,board_size=4): 
        board = Game_logic.Board(board_size)
        moves_count = 0
        while not board.game_over:
            board_tensor = Brain.convert_to_input(board.current_board_state, board_size)
            with torch.no_grad():
                move_probabilities = self.forward(board_tensor)
            move = torch.multinomial(move_probabilities, 1).item()
            moves = ["up", "down", "left", "right"]
            move = moves[move]
            moved =board.move(move)
            if moved:
                reward = evaluation_function(board.current_board_state)
                moves_count += 1
            else:
                reward = -1*abs(evaluate_board(board.current_board_state)) +3
            self.optimize(board_tensor,reward,move,learning_rate,gamma)
            
            if visualize:
                print('\033[2J\033[H')
                print(f"Move #{moves_count}")
                print(f"Move: {move}")
                print(f"Score: {board.score}")
                print("\nCurrent Board:")
                print(board)
                print(move_probabilities)
                print(f"Reward: {reward}")
                time.sleep(0.5)
        print("Game Over!")
    def learn_from_games(self ,num_games,learning_rate=0.01,gamma=0.9,evaluation_function=evaluate_board, visualize=False,board_size=4): 
        for game in range(num_games):
            self.learn(learning_rate,gamma,evaluation_function, visualize,board_size)
MC = PolicyNetwork()
MC.learn_from_games(10,learning_rate=0.001,gamma=0.9,evaluation_function=evaluate_board, visualize=True,board_size=4)
        






