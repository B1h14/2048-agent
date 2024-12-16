from shutil import move
from warnings import simplefilter
import Game_logic
import random
import copy
import Brain
import torch

class MonteCarloAgent:
    """An agent that uses Monte Carlo simulation with reinforcement learning to play 2048."""
    
    def __init__(self, game, num_simulations=10):
        """Initialize the Monte Carlo agent.
        
        Args:
            game: The 2048 game instance
            num_simulations: Number of simulations to run for each move evaluation
        """
        self.game = game
        self.num_simulations = num_simulations
        self.nn = Brain.NN()

    def get_best_move(self):
        """Determine the best move using Monte Carlo simulation.
        
        Returns:
            The move that yields the highest average score across simulations
        """
        moves = self.game.get_possible_actions()
        best_move = None
        best_score = 0
        
        for move in moves:
            total_score = 0
            for _ in range(self.num_simulations):
                simulated_game = copy.deepcopy(self.game)
                if simulated_game.move(move):
                    score = self.simulate_random_game(simulated_game)
                    total_score += score

            average_score = total_score / self.num_simulations
            if average_score > best_score:
                best_score = average_score
                best_move = move

        return best_move

    def simulate_random_game(self, game):
        """Simulate a game until completion using neural network-guided moves.
        
        Args:
            game: The game instance to simulate
            
        Returns:
            The final score of the simulated game
        """
        while not game.game_over:
            if not game.get_possible_actions():
                game.end_game()
            else:
                board_input = Brain.convert_to_input(game.current_board_state, 4)
                probabilities = self.nn.forward(board_input)
                move = Brain.perform_random_action(game, probabilities)
        return game.score

    def play_game_and_learn(self, num_simulations=10):
        """Play games while training the neural network on best moves found.
        
        Args:
            num_simulations: Number of games to simulate for training
        """
        max_tiles= []
        for game_num in range(num_simulations):
            agent = copy.deepcopy(self)
            moves_count = 0
            X_train = []
            Y_train = []
            while agent.game.get_possible_actions()  :
                best_move = agent.get_best_move()
                
                if best_move is None:
                    agent.game.end_game()
                    break
                
                best_move_index = Brain.convert_move_to_index(best_move)
                probabilities = torch.tensor([float(i == best_move_index) for i in range(4)])
                board_input = Brain.convert_to_input(Game_logic.copy_Board(agent.game.current_board_state), 4) # Create a copy of the board state to avoid modifying the original game state.agent.game.current_board_state, 4)
                X_train.append(board_input)
                Y_train.append(probabilities)
                
                print('\033[2J\033[H')  # Clear screen
                print(f"Game: {game_num}")
                print(f"Move #{moves_count}")
                print(f"Score: {agent.game.score}")
                print("\nCurrent Board:")
                print(agent.game)
                with torch.no_grad():
                    print("most probable move: ", Brain.convert_index_to_move( agent.nn.forward(board_input)))
                    print("probabilities: ",agent.nn.forward(board_input).detach().numpy())
                print(f"Best move: {best_move}")
                agent.game.move(best_move)
                moves_count += 1
            self.nn.train(X_train, Y_train)
            max_tiles.append(Game_logic.max_in_board(agent.game.current_board_state))
            print(f"Final score: {agent.game.score}")
            print(f"Final board:\n{agent.game}")
        
        print("max tiles: ", max_tiles)    
        torch.save(self.nn.state_dict(), "model.pth")

    def train(self, num_games):
        """Train the agent for a specified number of games.
        
        Args:
            num_games: Number of games to play for training
        """
        for game_num in range(num_games):
            print(f"Game: {game_num}")
            self.play_game_and_learn()

game = Game_logic.Board()
ai = MonteCarloAgent(game, 10)
ai.play_game_and_learn(50)
