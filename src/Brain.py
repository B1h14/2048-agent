from typing import List
import torch
import torch.nn as nn


class NN(nn.Module):
    """Neural network model for 2048 game AI.
    
    Architecture:
        - Input layer: 16 nodes (4x4 game board)
        - Hidden layer 1: 10 nodes
        - Hidden layer 2: 10 nodes
        - Output layer: 4 nodes (possible moves)
    """
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 4)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor representing the game board state
            
        Returns:
            Probability distribution over possible moves
        """
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.softmax(x)
        return x


def convert_to_input(board_list: List[List[int]], board_size: int) -> torch.Tensor:
    """Convert 2D board state to 1D input tensor.
    
    Args:
        board_list: 2D list representing the game board
        board_size: Size of the board (width/height)
        
    Returns:
        Flattened tensor representation of the board
    """
    flattened = [board_list[i // board_size][i % board_size] 
                 for i in range(board_size ** 2)]
    return torch.tensor(flattened, dtype=torch.float32)


def sort_args(values: List[float]) -> List[int]:
    """Sort indices by their corresponding values in descending order.
    
    Args:
        values: List of values to sort
        
    Returns:
        List of indices sorted by their values
    """
    indexed_values = [(val, idx) for idx, val in enumerate(values)]
    indexed_values.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in indexed_values]


def determine_action(board, model_output: List[float]) -> None:
    """Determine and perform the next action based on model output.
    
    Args:
        board: Game board object
        model_output: Model's predicted move probabilities
    """
    sorted_actions = sort_args(model_output)
    for action in sorted_actions:
        if perform_action(action, board):
            break
        if action == sorted_actions[-1]:
            board.end_game()


def perform_action(action: int, board) -> bool:
    """Perform the specified action on the board.
    
    Args:
        action: Integer representing the action (0-3)
        board: Game board object
        
    Returns:
        True if action was valid, False otherwise
    """
    actions = {
        0: board.down,
        1: board.up,
        2: board.left,
        3: board.right
    }
    return actions.get(action, lambda: True)()


def use_ai(board, board_size: int, model: NN) -> None:
    """Use AI model to make a move on the board.
    
    Args:
        board: Game board object
        board_size: Size of the board (width/height)
        model: Neural network model to use for prediction
    """
    board_tensor = convert_to_input(board.B, board_size)
    move_probabilities = model.forward(board_tensor)
    determine_action(board, move_probabilities)