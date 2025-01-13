import time
import math
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import Game_logic
import Brain


class PolicyNetwork(nn.Module):
    """Neural Network for predicting move probabilities."""
    
    def __init__(self, input_size=16, output_size=4):
        """Initialize the PolicyNetwork.
        
        Args:
            input_size: Size of input layer (default: 16)
            output_size: Number of output actions (default: 4)
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)
        
        self.fc1 = nn.Linear(64 * 1 * 1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x: Input tensor
        
        Returns:
            Softmax probabilities for actions
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return torch.softmax(self.fc4(x), dim=-1)

    def save(self, filename):
        """Save model weights.
        
        Args:
            filename: Path to save model
        """
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        """Load model weights.
        
        Args:
            filename: Path to load model from
        """
        self.load_state_dict(torch.load(filename))

    def optimize(self, board_input, reward, chosen_move=None, alpha=0.01, gamma=0.9):
        """Optimize network parameters.
        
        Args:
            board_input: Current board state
            reward: Reward for the move
            chosen_move: Selected move
            alpha: Learning rate
            gamma: Discount factor
        """
        output = self.forward(board_input).squeeze()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        if chosen_move is not None:
            chosen_move_index = Brain.convert_move_to_index(chosen_move)
            selected_component = output[chosen_move_index]
            selected_component.backward()
            
            with torch.no_grad():
                for param in self.parameters():
                    param += alpha * gamma * reward * param.grad / selected_component
                    param.grad.zero_()  # Reset gradients after update

    def learn(self, learning_rate=0.001, gamma=0.9, 
              evaluation_function=Game_logic.evaluate_board, visualize=False, board_size=4):
        """Learn from a single game.
        
        Args:
            learning_rate: Network learning rate
            gamma: Discount factor
            evaluation_function: Function to evaluate board state
            visualize: Whether to display game progress
            board_size: Size of game board
        """
        board = Game_logic.Board(board_size)
        moves_count = 0
        
        while board.get_possible_actions():
            board_tensor = Brain.convert_to_input_2D(board.current_board_state, board_size)
            
            with torch.no_grad():
                move_probabilities = self.forward(board_tensor).squeeze()
            
            move = torch.multinomial(move_probabilities, 1).item()
            moves = ["up", "down", "left", "right"]
            move = moves[move]
            
            moved = board.move(move)
            
            if moved:
                reward = evaluation_function(board.current_board_state)
                moves_count += 1
            else:
                reward = -1
            
            self.optimize(board_tensor, reward, move, learning_rate, gamma)
            
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

    def learn_from_games(self, num_games, learning_rate=0.01, gamma=0.9, 
                          evaluation_function=Game_logic.evaluate_board, visualize=False, board_size=4):
        """Learn from multiple games.
        
        Args:
            num_games: Number of games to play
            learning_rate: Network learning rate
            gamma: Discount factor
            evaluation_function: Function to evaluate board state
            visualize: Whether to display game progress
            board_size: Size of game board
        """
        for _ in range(num_games):
            self.learn(learning_rate, gamma, evaluation_function, visualize, board_size)