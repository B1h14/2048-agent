import random as rd
from typing import List, Optional
P = 0.01
def Random_Spawn_Position(Board_Size):
    l = rd.randint(0,15)
    i , j = l//Board_Size , l%Board_Size
    R = rd.random()
    Value = int(R>P) *2 + int(R<=P)*4
    return i,j,Value
def Spawn(board,Board_size):
    while True :
        i , j ,Value = Random_Spawn_Position(Board_size)
        if board[i][j] ==0 :
            board[i][j] = Value
            break
def copy_Board(board_list):
    B  = [ [ board_list[j][i] for i in range(len(board_list))] for j in range(len(board_list))]
    return B
class Board():
    def __init__(self,size = 4):
        self.current_board_state = [ [ 0 for i in range(size)] for j in range(size)]
        self.Board_states = []
        self.Board_states+=copy_Board(self.current_board_state)
        self.size = size
        Spawn(self.current_board_state,size)
        self.score = 0
        self.game_over = False
    def __str__(self):
        Output = ""
        for k in range(self.size):
            for l in range(self.size):
                Output+=str(self.current_board_state[k][l])
                Output+=' '
            Output+="\n"
        return Output
    def check_right(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[i][n-l] == 0 and self.current_board_state[i][n-l-1] !=0:
                        Action = True
                        break
                    elif self.current_board_state[i][n-l-1] !=0 and self.current_board_state[i][n-l]==self.current_board_state[i][n-l-1] :
                        Action = True
                        break
        if(Action):
            return True
        else:
            return False
    def check_down(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[n-l][i] == 0 and self.current_board_state[n-l-1][i] !=0:
                        Action = True
                        break
                    elif self.current_board_state[n-l-1][i] !=0 and self.current_board_state[n-l][i]==self.current_board_state[n-l-1][i] :
                        Action = True
                        break
        if(Action):
            return True
        else:
            return False
    def check_up(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[l-1][i] == 0 and self.current_board_state[l][i] !=0:
                        Action = True
                        break
                    elif self.current_board_state[l][i] !=0 and self.current_board_state[l-1][i]==self.current_board_state[l][i] :
                        Action = True
                        break
        if(Action):
            return True
        else:
            return False
    def check_left(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[i][l-1] == 0 and self.current_board_state[i][l] !=0:
                        Action = True
                        break
                    elif self.current_board_state[i][l] !=0 and self.current_board_state[i][l-1]==self.current_board_state[i][l] :
                        Action = True
                        break
        if(Action):
            return True
        else:
            return False
    def down(self):
        Action = False
        self.Board_states.append(copy_Board(self.current_board_state))

        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[n-l][i] == 0 and self.current_board_state[n-l-1][i] !=0:
                        self.current_board_state[n-l][i] = self.current_board_state[n-l-1][i]
                        self.current_board_state[n-l-1][i] = 0
                        Action = True
                    elif self.current_board_state[n-l-1][i] !=0 and self.current_board_state[n-l][i]==self.current_board_state[n-l-1][i] :
                        self.current_board_state[n-l][i]*=2
                        self.score+=2*self.current_board_state[n-l-1][i]
                        self.current_board_state[n-l-1][i] = 0
                        Action = True
        if(Action):
            Spawn(self.current_board_state,n)
            return True
        else:
            return False
    def right(self):
        self.Board_states.append(copy_Board(self.current_board_state))
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[i][n-l] == 0 and self.current_board_state[i][n-l-1] !=0:
                        self.current_board_state[i][n-l] = self.current_board_state[i][n-l-1]
                        self.current_board_state[i][n-l-1] = 0
                        Action = True
                    elif self.current_board_state[i][n-l-1] !=0 and self.current_board_state[i][n-l]==self.current_board_state[i][n-l-1] :
                        self.current_board_state[i][n-l]*=2
                        self.score+=2*self.current_board_state[i][n-l-1]
                        self.current_board_state[i][n-l-1] = 0
                        Action = True
        if(Action):
            Spawn(self.current_board_state,n)
            return True
        else:
            return False
    def up(self):
        self.Board_states.append(copy_Board(self.current_board_state))
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[l-1][i] == 0 and self.current_board_state[l][i] !=0:
                        self.current_board_state[l-1][i] = self.current_board_state[l][i]
                        self.current_board_state[l][i] = 0
                        Action = True
                    elif self.current_board_state[l][i] !=0 and self.current_board_state[l-1][i]==self.current_board_state[l][i] :
                        self.current_board_state[l-1][i]*=2
                        self.score+=2*self.current_board_state[l-1][i]
                        self.current_board_state[l][i] = 0
                        Action = True
        if(Action):
            Spawn(self.current_board_state,n)
            return True
        else:
            return False
    def left(self):
        Action = False
        self.Board_states.append(copy_Board(self.current_board_state))
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.current_board_state[i][l-1] == 0 and self.current_board_state[i][l] !=0:
                        self.current_board_state[i][l-1] = self.current_board_state[i][l]
                        self.current_board_state[i][l] = 0
                        Action = True
                    elif self.current_board_state[i][l] !=0 and self.current_board_state[i][l-1]==self.current_board_state[i][l] :
                        self.current_board_state[i][l-1]*=2
                        self.score+=2*self.current_board_state[i][l]
                        self.current_board_state[i][l] = 0
                        Action = True
        if(Action):
            Spawn(self.current_board_state,n)
            return True
        else:
            return False
    def end_game(self):
        self.game_over = True
    def get_possible_actions(self):
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
    def go_back(self):
        self.current_board_state = copy_Board(self.Board_states.pop(-1))
    def set_state(self, state):
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

def perform_Action(move,Board) :
    if move == "up" :
        return Board.up()
    elif move == "down" :
        return Board.down()
    elif move == "left" :
        return Board.left()
    elif move == "right" :
        return Board.right()
    else :
        return Board.end_game()  
def max_in_board(board_list: List[List[int]]) -> int:
    """Find the maximum value in the game board.
    
    Args:
        board_list: 2D list representing the game board
        
    Returns:
        Maximum value found
    """
    return max(max(row) for row in board_list)