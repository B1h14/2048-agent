import copy
import random

import torch

import Game_logic
import Brain


class MonteCarloAgent:
    """
    A Monte Carlo simulation agent for playing the 2048 game.
    
    This agent uses Monte Carlo tree search to evaluate and select the best move
    by running multiple random game simulations from each possible move.
    
    Attributes:
        game (Game_logic.Board): The current game instance
        num_simulations (int): Number of random game simulations to run per move
    """
    
    def __init__(self, game=None, num_simulations=1000):
        """
        Initialize the Monte Carlo agent.
        
        Args:
            game (Game_logic.Board, optional): The 2048 game instance. 
                Defaults to a new 4x4 board if not provided.
            num_simulations (int, optional): Number of simulations to run 
                for each move evaluation. Defaults to 1000.
        """
        self.game = game or Game_logic.Board(4)
        self.num_simulations = num_simulations

    def get_best_move(self):
        """
        Determine the optimal move using Monte Carlo simulation.
        
        Evaluates each possible move by running multiple random game simulations
        and selecting the move with the highest average score.
        
        Returns:
            str or None: The best move ('up', 'down', 'left', 'right'), 
                or None if no valid moves exist
        """
        possible_moves = self.game.get_possible_actions()
        
        if not possible_moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in possible_moves:
            total_score = 0
            
            for _ in range(self.num_simulations):
                # Create a deep copy to prevent modifying the original game state
                simulated_game = copy.deepcopy(self.game)
                
                if simulated_game.move(move):
                    score = self.simulate_random_game(simulated_game)
                    total_score += score

            # Calculate average score for the current move
            average_score = total_score / self.num_simulations
            
            if average_score > best_score:
                best_score = average_score
                best_move = move

        return best_move

    def simulate_random_game(self, game):
        """
        Simulate a game to completion using random moves.
        
        Plays out a game from the given state using random move selection
        until the game is over.
        
        Args:
            game (Game_logic.Board): The game instance to simulate
        
        Returns:
            float: The final score of the simulated game
        """
        score = 0
        
        while not game.game_over:
            if not game.get_possible_actions():
                game.end_game()
                break
            
            actions = ['up', 'down', 'left', 'right']
            move = random.choice(actions)
            game.move(move)
            
            score += Game_logic.evaluate_board(game.current_board_state)
        
        return score