import random as rd

P = 0.25
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

class Board():
    def __init__(self,size = 4):
        self.B = [ [ 0 for i in range(size)] for j in range(size)]
        self.size = size
        Spawn(self.B,size)
        self.score = 0
        self.game_over = False
    def __str__(self):
        Output = ""
        for k in range(self.size):
            for l in range(self.size):
                Output+=str(self.B[k][l])
                Output+=' '
            Output+="\n"
        return Output
    def down(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.B[n-l][i] == 0 and self.B[n-l-1][i] !=0:
                        self.B[n-l][i] = self.B[n-l-1][i]
                        self.B[n-l-1][i] = 0
                        Action = True
                    elif self.B[n-l-1][i] !=0 and self.B[n-l][i]==self.B[n-l-1][i] :
                        self.B[n-l][i]*=2
                        self.score+=2*self.B[n-l-1][i]
                        self.B[n-l-1][i] = 0
                        Action = True
        if(Action):
            Spawn(self.B,n)
            return True
        else:
            return False
    def right(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.B[i][n-l] == 0 and self.B[i][n-l-1] !=0:
                        self.B[i][n-l] = self.B[i][n-l-1]
                        self.B[i][n-l-1] = 0
                        Action = True
                    elif self.B[i][n-l-1] !=0 and self.B[i][n-l]==self.B[i][n-l-1] :
                        self.B[i][n-l]*=2
                        self.score+=2*self.B[i][n-l-1]
                        self.B[i][n-l-1] = 0
                        Action = True
        if(Action):
            Spawn(self.B,n)
            return True
        else:
            return False
    def up(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.B[l-1][i] == 0 and self.B[l][i] !=0:
                        self.B[l-1][i] = self.B[l][i]
                        self.B[l][i] = 0
                        Action = True
                    elif self.B[l][i] !=0 and self.B[l-1][i]==self.B[l][i] :
                        self.B[l-1][i]*=2
                        self.score+=2*self.B[l-1][i]
                        self.B[l][i] = 0
                        Action = True
        if(Action):
            Spawn(self.B,n)
            return True
        else:
            return False
    def left(self):
        Action = False
        n = self.size
        for i in range(n):
            for k in range(1,n):
                for l in range(k,0,-1):
                    if self.B[i][l-1] == 0 and self.B[i][l] !=0:
                        self.B[i][l-1] = self.B[i][l]
                        self.B[i][l] = 0
                        Action = True
                    elif self.B[i][l] !=0 and self.B[i][l-1]==self.B[i][l] :
                        self.B[i][l-1]*=2
                        self.score+=2*self.B[i][l]
                        self.B[i][l] = 0
                        Action = True
        if(Action):
            Spawn(self.B,n)
            return True
        else:
            return False
    def end_game(self):
        self.game_over = True
C = Board()
C.up()
C.down()
C.left()
C.up()
C.right()
C.down()

print(C)
