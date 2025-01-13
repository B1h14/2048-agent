import random as rd
from typing import List, Optional, Tuple
import math

# Probability of spawning a 4 instead of a 2
P = 0.1

def Random_Spawn_Position(Board_Size: int) -> Tuple[int, int, int]:
    """
    Generate a random spawn position and value for the game board.
    
    Args:
        Board_Size (int): Size of the game board
    
    Returns:
        Tuple[int, int, int]: Row, column, and spawn value
    """
    l = rd.randint(0, 15)
    i, j = l // Board_Size, l % Board_Size
    R = rd.random()
    Value = int(R > P) * 2 + int(R <= P) * 4
    return i, j, Value

def Spawn(board: List[List[int]], Board_size: int) -> None:
    """
    Spawn a new tile on an empty position of the board.
    
    Args:
        board (List[List[int]]): Game board
        Board_size (int): Size of the game board
    """
    while True:
        i, j, Value = Random_Spawn_Position(Board_size)
        if board[i][j] == 0:
            board[i][j] = Value
            break

def copy_Board(board_list: List[List[int]]) -> List[List[int]]:
    """
    Create a deep copy of the game board.
    
    Args:
        board_list (List[List[int]]): Original board to be copied
    
    Returns:
        List[List[int]]: A new board with the same values as the input board
    """
    B = [[board_list[j][i] for i in range(len(board_list))] for j in range(len(board_list))]
    return B

class Board:
    def __init__(self, size: int = 4):
        """
        Initialize a new 2048 game board.
        
        Args:
            size (int, optional): Size of the game board. Defaults to 4.
        """
        self.current_board_state = [[0 for _ in range(size)] for _ in range(size)]
        self.Board_states: List[List[List[int]]] = []
        self.Board_states += copy_Board(self.current_board_state)
        self.size = size
        Spawn(self.current_board_state, size)
        self.score = 0
        self.game_over = False

    def __str__(self) -> str:
        """
        Generate a string representation of the game board.
        
        Returns:
            str: Formatted board state with tiles separated by spaces
        """
        Output = ""
        for k in range(self.size):
            for l in range(self.size):
                Output += str(self.current_board_state[k][l])
                Output += ' '
            Output += "\n"
        return Output

    def check_up(self):
        """
        Check if the 'up' move is viable.

        Returns:
            bool: True if the 'up' move is viable, False otherwise.
        """
        board = self.current_board_state

        for col in range(self.size):
            for row in range(1, self.size):
                if board[row][col] != 0:
                    if board[row - 1][col] == 0 or board[row - 1][col] == board[row][col]:
                        return True
        return False

    def check_down(self):
        """
        Check if the 'down' move is viable.

        Returns:
            bool: True if the 'down' move is viable, False otherwise.
        """
        board = self.current_board_state

        for col in range(self.size):
            for row in range(self.size - 1):  # Exclude the last row
                if board[row][col] != 0:
                    if board[row + 1][col] == 0 or board[row + 1][col] == board[row][col]:
                        return True
        return False

    def check_left(self):
        """
        Check if the 'left' move is viable.

        Returns:
            bool: True if the 'left' move is viable, False otherwise.
        """
        board = self.current_board_state

        for row in range(self.size):
            for col in range(1, self.size):  # Start from the second column
                if board[row][col] != 0:
                    if board[row][col - 1] == 0 or board[row][col - 1] == board[row][col]:
                        return True
        return False

    def check_right(self):
        """
        Check if the 'right' move is viable.

        Returns:
            bool: True if the 'right' move is viable, False otherwise.
        """
        board = self.current_board_state

        for row in range(self.size):
            for col in range(self.size - 1):  # Exclude the last column
                if board[row][col] != 0:
                    if board[row][col + 1] == 0 or board[row][col + 1] == board[row][col]:
                        return True
        return False

    def down(self) -> bool:
        """
        Move tiles downward and merge tiles with the same value.
        
        Returns:
            bool: True if the board state changed, False otherwise
        """
        Action = False
        self.Board_states.append(copy_Board(self.current_board_state))
        n = self.size
        
        for i in range(n):
            # Extract non-zero tiles in the column
            row = [self.current_board_state[j][i] for j in range(n) if self.current_board_state[j][i] != 0]
            row += [0] * (n - len(row))
            
            # Merge tiles
            for j in range(len(row) - 1, 0, -1):
                if row[j] == row[j - 1]:
                    row[j] *= 2
                    row[j - 1] = 0
                    self.score += 2 * row[j - 1]
            
            # Remove zeros and pad
            row = [row[j] for j in range(len(row) - 1, -1, -1) if row[j] != 0]
            row += [0] * (n - len(row))
            
            # Update board state
            for j in range(n):
                if self.current_board_state[j][i] != row[n - j - 1]:
                    Action = True
                self.current_board_state[j][i] = row[n - j - 1]
        
        if Action:
            Spawn(self.current_board_state, n)
            return True
        
        return False

    def right(self) -> bool:
        """
        Move tiles to the right and merge tiles with the same value.
        
        Returns:
            bool: True if the board state changed, False otherwise
        """
        Action = False
        self.Board_states.append(copy_Board(self.current_board_state))
        n = self.size
        
        for i in range(n):
            # Extract non-zero tiles in the row
            row = [self.current_board_state[i][j] for j in range(n) if self.current_board_state[i][j] != 0]
            row += [0] * (n - len(row))
            
            # Merge tiles
            for j in range(len(row) - 1, 0, -1):
                if row[j] == row[j - 1]:
                    row[j - 1] *= 2
                    row[j] = 0
                    self.score += row[j - 1]
            
            # Remove zeros and pad
            row = [row[j] for j in range(len(row) - 1, -1, -1) if row[j] != 0]
            row += [0] * (n - len(row))
            
            # Update board state
            for j in range(n):
                if self.current_board_state[i][j] != row[n - j - 1]:
                    Action = True
                self.current_board_state[i][j] = row[n - j - 1]
        
        if Action:
            Spawn(self.current_board_state, n)
            return True
        
        return False

    def up(self) -> bool:
        """
        Move tiles upward and merge tiles with the same value.
        
        Returns:
            bool: True if the board state changed, False otherwise
        """
        Action = False
        self.Board_states.append(copy_Board(self.current_board_state))
        n = self.size
        
        for i in range(n):
            # Extract non-zero tiles in the column
            row = [self.current_board_state[j][i] for j in range(n) if self.current_board_state[j][i] != 0]
            row += [0] * (n - len(row))
            
            # Merge tiles
            for j in range(len(row) - 1):
                if row[j] == row[j + 1]:
                    row[j + 1] *= 2
                    self.score += 2 * row[j + 1]
                    row[j] = 0
            
            # Remove zeros and pad
            row = [row[j] for j in range(len(row)) if row[j] != 0]
            row += [0] * (n - len(row))
            
            # Update board state
            for j in range(n):
                if self.current_board_state[j][i] != row[j]:
                    Action = True
                self.current_board_state[j][i] = row[j]
        
        if Action:
            Spawn(self.current_board_state, n)
            return True
        
        return False

    def left(self) -> bool:
        """
        Move tiles to the left and merge tiles with the same value.
        
        Returns:
            bool: True if the board state changed, False otherwise
        """
        Action = False
        self.Board_states.append(copy_Board(self.current_board_state))
        n = self.size
        
        for i in range(n):
            # Extract non-zero tiles in the row
            row = [self.current_board_state[i][j] for j in range(n) if self.current_board_state[i][j] != 0]
            row += [0] * (n - len(row))
            
            # Merge tiles
            for j in range(len(row) - 1):
                if row[j] == row[j + 1]:
                    row[j + 1] *= 2
                    row[j] = 0
                    self.score += row[j + 1]
            
            # Remove zeros and pad
            row = [row[j] for j in range(len(row)) if row[j] != 0]
            row += [0] * (n - len(row))
            
            # Update board state
            for j in range(n):
                if self.current_board_state[i][j] != row[j]:
                    Action = True
                self.current_board_state[i][j] = row[j]
        
        if Action:
            Spawn(self.current_board_state, n)
            return True
        
        return False  

    def end_game(self) -> None:
        """
        Mark the game as over.
        
        Sets the game_over flag to True, indicating the game has ended.
        """
        self.game_over = True

    def get_possible_actions(self) -> List[str]:
        """
        Determine all viable moves for the current board state.
        
        Returns:
            List[str]: A list of possible moves ('right', 'left', 'up', 'down')
        """
        possible_actions = []
        
        if self.check_right():
            possible_actions.append("right")
        
        if self.check_left():
            possible_actions.append("left")
        
        if self.check_up():
            possible_actions.append("up")
        
        if self.check_down():
            possible_actions.append("down")
        
        return possible_actions

    def go_back(self) -> None:
        """
        Revert the board to the previous state.
        
        Restores the board to the last saved state from Board_states.
        """
        self.current_board_state = copy_Board(self.Board_states.pop(-1))

    def set_state(self, state: List[List[int]]) -> None:
        """
        Set the current board state to a specific configuration.
        
        Args:
            state (List[List[int]]): The board state to set
        """
        self.current_board_state = copy_Board(state)

        def move(self, move_str: str) -> bool:
            """Perform a move on the board based on the input string.
            
            Args:
                move_str: String representing the move ('up', 'down', 'left', 'right')
                
            Returns:
                bool: True if the move was valid and performed, False otherwise
            """
            if move_str == "up":
                return self.up()
            elif move_str == "down":
                return self.down()
            elif move_str == "left":
                return self.left()
            elif move_str == "right":
                return self.right()
            return False

def max_in_board(board_list: List[List[int]]) -> int:
    """Find the maximum value in the game board.
    
    Args:
        board_list: 2D list representing the game board
        
    Returns:
        Maximum value found
    """
    return max(max(row) for row in board_list)

def monotomy(board: List[List[int]]) -> int:
    """
    Calculate the monotony of the game board.
    
    Monotony measures the consistency of tile values in a specific direction.
    
    Args:
        board (List[List[int]]): Game board to evaluate
    
    Returns:
        int: Monotony distance (number of consecutive monotonically decreasing tiles)
    """
    distance = 0
    consecutives = True
    board_size = len(board)
    
    for k in range(board_size**2 - 1):
        i = k // board_size
        j = k % board_size
        j = j * (i % 2) + (board_size - 1 - j) * (1 - i % 2)
        
        l = (k + 1) // board_size
        m = (k + 1) % board_size
        m = m * (l % 2) + (board_size - 1 - m) * (1 - l % 2)
        
        if board[i][j] > board[l][m] and consecutives:
            distance += 1
        else:
            consecutives = False
            break
    
    return distance

def smoothness(board: List[List[int]]) -> float:
    """
    Calculate board smoothness by measuring the logarithmic differences 
    between adjacent tiles.
    
    Args:
        board (List[List[int]]): Game board to evaluate
    
    Returns:
        float: Smoothness score (lower is smoother)
    """
    smoothness_val = 0
    board_size = len(board)
    
    for i in range(board_size - 1):
        for j in range(board_size - 1):
            if board[i][j] != 0 and board[i][j + 1] != 0:
                smoothness_val += abs(math.log2(board[i][j]) - math.log2(board[i][j + 1]))**2
            
            if board[i][j] != 0 and board[i + 1][j] != 0:
                smoothness_val += abs(math.log2(board[i][j]) - math.log2(board[i + 1][j]))**2
    
    return smoothness_val

def evaluate_board(board: List[List[int]], a: int = 10, b: int = 1, c: int = 1) -> float:
    """
    Evaluate the game board state using multiple criteria.
    
    Args:
        board (List[List[int]]): Game board to evaluate
        a (int, optional): Weight for max tile placement. Defaults to 10.
        b (int, optional): Weight for empty tiles. Defaults to 1.
        c (int, optional): Weight for monotony. Defaults to 1.
    
    Returns:
        float: Comprehensive board evaluation score
    """
    zero_count = sum(row.count(0) for row in board)
    
    score = (
        math.log2(max_in_board(board)) * (a * (board[0][0] == max_in_board(board))) +
        b * zero_count +
        c * monotomy(board) -
        0.3 * smoothness(board)
    )
    
    return score