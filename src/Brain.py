from typing import List
import torch
import torch.nn as nn
import math

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
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=0)
        self.ReLU = nn.ReLU()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor representing the game board state
            
        Returns:
            Probability distribution over possible moves
        """
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.ReLU(self.fc3(x))
        x = self.ReLU(self.fc4(x))

        x = self.softmax(x)
        return x
    
    def train(self, X, Y, learning_rate=0.01):
        """Train the network on batches of data.
        
        Args:
            X: List of input tensors
            Y: List of target tensors
            learning_rate: Learning rate for optimization
        """
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Convert lists to tensors and stack them
        X_batch = torch.stack(X)
        Y_batch = torch.stack(Y)
        
        optimizer.zero_grad()
        outputs = self.forward(X_batch)
        
        # Apply log to outputs for KL divergence
        log_outputs = torch.log(outputs + 1e-10)  # Add small constant for numerical stability
        
        loss = criterion(log_outputs, Y_batch)
        loss.backward()
        optimizer.step()
        
        return loss.item()

def convert_to_input(board_list: List[List[int]], board_size: int) -> torch.Tensor:
    """Convert 2D board state to 1D input tensor.
    
    Args:
        board_list: 2D list representing the game board
        board_size: Size of the board (width/height)
        
    Returns:
        Flattened tensor representation of the board
    """
    flattened = [ math.log2(board_list[i // board_size][i % board_size]) if board_list[i // board_size][i % board_size] != 0 else board_list[i // board_size][i % board_size] 
                 for i in range(board_size ** 2)]
    return torch.tensor(flattened, dtype=torch.float32)
def convert_to_input_2D(board_list: List[List[int]], board_size: int) -> torch.Tensor:
    """Convert 2D board state to 1D input tensor.
    
    Args:
        board_list: 2D list representing the game board
        board_size: Size of the board (width/height)
        
    Returns:
        Flattened tensor representation of the board
    """
    flattened = [[ math.log2(board_list[i // board_size][i % board_size]) if board_list[i // board_size][i % board_size] != 0 else board_list[i // board_size][i % board_size] ]
                 for i in range(board_size ** 2)]
    return torch.tensor(flattened, dtype=torch.float32).view(1, 1, board_size, board_size)

def convert_move_to_index(move: str) -> int:
    """Convert move string to index.
    
    Args:
        move: String representing the move (up, down, left, right)
        
    Returns:
        Integer representing the index of the move
    """
    moves = ["up", "down", "left", "right"]
    return moves.index(move)

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

def determine_action(board, model_output) -> None:
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
        0: board.up,
        1: board.down,
        2: board.left,
        3: board.right
    }
    return actions.get(action, lambda: True)()

def perform_random_action(board , probabilities) :
    action = torch.multinomial(probabilities, 1).item()
    actions = {
        0: board.up,
        1: board.down,
        2: board.left,
        3: board.right
    }
    return actions.get(action, lambda: True)()

def convert_index_to_move(probabilities: torch.Tensor) -> str:
    """Convert probability vector to most probable move string.
    
    Args:
        probabilities: Tensor or list of probabilities for each move
        
    Returns:
        String representing the most probable move (up, down, left, right)
    """
    moves = ["up", "down", "left", "right"]
    max_index = torch.argmax(probabilities).item() if isinstance(probabilities, torch.Tensor) else probabilities.index(max(probabilities))
    return moves[max_index]

def use_ai(board, board_size: int, model: NN) -> None:
    """Use AI model to make a move on the board.
    
    Args:
        board: Game board object
        board_size: Size of the board (width/height)
        model: Neural network model to use for prediction
    """
    board_tensor = convert_to_input(board.current_board_state, board_size)
    move_probabilities = model.forward(board_tensor).detach().numpy()
    determine_action(board, move_probabilities)