import dis
from itertools import count
from re import M
import time
from turtle import distance

from torch import zero_
import Game_logic
from typing import List, Optional
import math
def max_in_board(board_list: List[List[int]]) -> int:
    """Find the maximum value in the game board.
    
    Args:
        board_list: 2D list representing the game board
        
    Returns:
        Maximum value found
    """
    return max(max(row) for row in board_list)
def evaluation_function_1(board):
    return sum(sum(row) for row in board) + max_in_board(board)**2
def evaluation_function_2(board): 
    distance = 0
    for i in range(len(board)-1):
        for j in range(len(board)-1):
            if board[i][j] != 0:
                distance += abs(board[i][j+1] - board[i][j] )**2
                distance += abs(board[i+1][j] - board[i][j] )**2

    return  max_in_board(board)**2 -distance
def evaluation_function_3(board): 
    distance = 0
    zero_count = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                zero_count += 1
    for i in range(len(board)-1):
        for j in range(len(board)-1):
            distance += abs(board[i][j+1] - board[i][j] )**2
            distance += abs(board[i+1][j] - board[i][j] )**2
    return  max_in_board(board)**zero_count -distance
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
def evaluation_function_4(board,a=10,b=1,c=1):
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
def evaluate(board: list[list[int]]) -> float:
    """
    Evaluate the current game board state using weighted heuristics.
    Board is a 4x4 list of integers.
    """
    empty_cells = count_empty_cells(board)
    max_tile = max(max(row) for row in board)
    smoothness = calculate_smoothness(board)
    monotonicity = calculate_monotonicity(board)

    # Weighted sum of features
    score = (
        3.0 * empty_cells +        # Prioritize empty cells
        1.5 * math.log2(max_tile) + # Reward larger tiles
        1.0 * monotonicity +       # Encourage monotonic rows/columns
        -2.0 * smoothness          # Penalize unsmooth boards
    )
    return score

# Supporting Functions
def count_empty_cells(board: list[list[int]]) -> int:
    """Count empty cells on the board."""
    return sum(row.count(0) for row in board)

def calculate_smoothness(board: list[list[int]]) -> int:
    """Penalize boards with adjacent tiles of different values."""
    smoothness = 0
    for i in range(4):
        for j in range(3):
            if board[i][j] != 0 and board[i][j] == board[i][j+1]:
                smoothness -= abs(board[i][j] - board[i][j+1])
            if board[j][i] != 0 and board[j][i] == board[j+1][i]:
                smoothness -= abs(board[j][i] - board[j+1][i])
    return smoothness

def calculate_monotonicity(board: list[list[int]]) -> int:
    """Reward boards with monotonic rows and columns."""
    score = 0
    
    # Row-wise monotonicity
    for row in board:
        score += row_monotonicity(row)
    
    # Column-wise monotonicity
    for col in zip(*board):
        score += row_monotonicity(col)
    
    return score

def row_monotonicity(row: list[int]) -> int:
    """Calculate row monotonicity (reward ordered sequences)."""
    increase = sum(row[i] >= row[i+1] for i in range(3))
    decrease = sum(row[i] <= row[i+1] for i in range(3))
    return max(increase, decrease)
class Minimax2048:
    def __init__(self, depth=3):
        self.depth = depth

    def evaluate(self, board):
        return evaluate(board)

    def minimax(self, game, depth, maximizing):
        if depth == 0 or game.game_over:
            return self.evaluate(game.current_board_state), None

        if maximizing:
            max_eval = float('-inf')
            best_move = None
            for move in ['up', 'down', 'left', 'right']:
                new_game = Game_logic.Board()
                new_game.current_board_state = Game_logic.copy_Board(game.current_board_state)  # Create a copy of the board state to avoid modifying the original game state.game.current_board_state()
                if new_game.move(move):
                    eval, _ = self.minimax(new_game, depth - 1, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for i in range(4):
                for j in range(4):
                    if game.current_board_state[i][j] == 0:
                        for tile in [2, 4]:
                            new_game = Game_logic.Board()
                            new_game.current_board_state = Game_logic.copy_Board(game.current_board_state)
                            new_game.current_board_state[i][j] = tile
                            eval, _ = self.minimax(new_game, depth - 1, True)
                            min_eval = min(min_eval, eval)
            return min_eval, None


def play_game(depth=3):
    game = Game_logic.Board()
    ai = Minimax2048(depth)
    moves_count = 0
    while not game.game_over:
        _, best_move = ai.minimax(game, ai.depth, True)
        if best_move:
            print('\033[2J\033[H')
            print(f"Move #{moves_count}")
            print(f"Score: {evaluation_function_4(game.current_board_state)}")
            print("\nCurrent Board:")
            print(game)
            possible_moves = game.get_possible_actions()
            game.move(best_move)
            print("possible moves")
            print(possible_moves)
            print(f"Move: {best_move}")
            time.sleep(1)
            moves_count += 1
        else: 
            print('\033[2J\033[H')
            print(f"Move #{moves_count}")
            print(f"Score: {game.score}")
            print("\nCurrent Board:")
            print(game)
            print("No valid moves available. Game Over!")
            break
    print("Game Over!")
def create_dataset(number_of_samples,depth=3,Board_Size=4):
    X_data = []
    Y_data = []
    max_tile = []
    for i in range(number_of_samples):
        game = Game_logic.Board()
        ai = Minimax2048(depth)
        while not game.game_over:
            _, best_move = ai.minimax(game, ai.depth, True)
            if len(game.get_possible_actions()) > 0 and best_move:
                X_data.append(Game_logic.copy_Board(game.current_board_state))
                Y_data.append(best_move)
                game.move(best_move)
            else: 
                max_tile.append(max_in_board(game.current_board_state))
                game.end_game()
                
        print(f"Sample {i} created. maximum tile: {max_in_board(game.current_board_state)} Board: \n {game}")
    f = open("X_data.csv","w")
    for _ in range(len(X_data)):
        for i in range(Board_Size):
            for j in range(Board_Size):
                f.write(str(X_data[_][i][j])+";")
        f.write("\n") 
    f.close()
    f = open("Y_data.csv","w")
    for _ in range(len(Y_data)):
        f.write(Y_data[_]+"\n")
    f.close()
    f= open("max_tile.csv","w")
    for _ in range(len(max_tile)):
        f.write(str(max_tile[_])+"\n")
    f.close()
    print("max tile reached: ", max(max_tile))
    print("done")
create_dataset(20,5)