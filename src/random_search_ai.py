import Game_logic
import random
import copy
import Brain
import torch
class MonteCarlo2048:
    def __init__(self, game, number_of_simulations=100):
        self.game = game
        self.number_of_simulations = number_of_simulations

    def get_best_move(self):
        moves = self.game.get_possible_actions()
        best_move = None
        best_score = 0
        for move in moves:
            total_score = 0

            for _ in range(self.number_of_simulations):
                simulated_game = copy.deepcopy(self.game)
                if simulated_game.move(move):
                    score = self.simulate_random_game(simulated_game)
                    total_score += score

            average_score = total_score / self.number_of_simulations
            if average_score > best_score:
                best_score = average_score
                best_move = move

        return best_move

    def simulate_random_game(self, game):
        """Simulates a game until completion using random moves."""
        while not game.game_over:
            if game.get_possible_actions() == []:
                game.end_game()
            else:
                move = random.choice(game.get_possible_actions())
                game.move(move)
        return game.score

class MonteCarlo2048_RL:
    def __init__(self, game, number_of_simulations=100):
        self.game = game
        self.number_of_simulations = number_of_simulations
        self.NN = Brain.NN()   

    def get_best_move(self):
        moves = self.game.get_possible_actions()
        best_move = None
        best_score = 0
        for move in moves:
            total_score = 0

            for _ in range(self.number_of_simulations):
                simulated_game = copy.deepcopy(self.game)
                if simulated_game.move(move):
                    score = self.simulate_random_game(simulated_game)
                    total_score += score

            average_score = total_score / self.number_of_simulations
            if average_score > best_score:
                best_score = average_score
                best_move = move

        return best_move

    def simulate_random_game(self, game):
        """Simulates a game until completion using random moves."""
        while not game.game_over:
            if game.get_possible_actions() == []:
                game.end_game()
            else:
                probabilities = self.NN.forward(Brain.convert_to_input(game.current_board_state,4))
                move = Brain.perform_random_action(game,probabilities)
        return game.score
    def play_game_and_learn(self):
        MCai = copy.deepcopy(self)
        while not MCai.game.get_possible_actions() == []:
            best_move = MCai.get_best_move()

            if best_move == None:
                MCai.game.end_game()
                break
            else:
                best_move_index = Brain.convert_move_to_index(best_move)
                probabilities = torch.tensor([float(i== best_move_index ) for i in range(4)])
                self.NN.train(Brain.convert_to_input(MCai.game.current_board_state,4),probabilities)
                print(f"Score: {MCai.game.score}")
                print("\nCurrent Board:")
                print(MCai.game)
                print("move: ",best_move)
                MCai.game.move(best_move)

        return (MCai.game.score , MCai.game.current_board_state)
    def learning(self,num_games):
        for _ in range(num_games):
            print("game: ",_)
            score , board = self.play_game_and_learn()
            print("final score: ",score)
            print("final board: ",board)
game = Game_logic.Board()
ai = MonteCarlo2048_RL(game,10)
ai.learning(10)
