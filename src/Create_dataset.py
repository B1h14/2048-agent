from openai import OpenAI
import Game_logic
import time
import ollama_test as ot
def get_ai_completion(client, messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return completion.choices[0].message.content

def get_chatgpt_action(client,Board,Board_size):
    messages = [
        {"role": "system", "content": "You are an AI playing 2048. Analyze the board and choose the best move. keep the biggest number in the corner always and try not to have small numbers trapped"},
        {"role": "user", "content": f"""Current board state:\n{str(Board)}\nChoose an action between: {Board.get_possible_actions()}\nAnswer with only one word in lowercase and make sure it is a valid action."""}
    ]
    move = get_ai_completion(client, messages).strip().lower()
    return move
def request_different_actions(client,Board,Board_size):
    content = " that move is not valid and ask for a different action between: up, down, left, right\nAnswer with only one word in lowercase and make sure it is a valid action."
    messages = [
        {"role": "system", "content": content}, 
    ]
    move = get_ai_completion(client, messages).strip().lower()
    return move   
def create_dataset(Board_Size,number_of_samples):
    api_key = input("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    Board = Game_logic.Board(Board_Size)
    X_data = []
    Y_data = []
    for i in range(number_of_samples):
        Board = Game_logic.Board(Board_Size)
        moves_count = 0
        while not(Board.game_over):
            print('\033[2J\033[H')
            print(f"Move #{moves_count}")
            print(f"Score: {Board.score}")
            print("\nCurrent Board:")
            print(Board)
            X = Board.current_board_state[:]
            X_data.append(X)
            move = get_chatgpt_action(client,Board,Board_Size)
            Y_data.append(move)


            
            # Execute the move
            moved = False
            if move == "up":
                moved = Board.up()
            elif move == "down":
                moved = Board.down()
            elif move == "left":
                moved = Board.left()
            elif move == "right":
                moved = Board.right()
                
            # Check if move was successful
                   
            # Check if game is over (no valid moves available)
            if not moved and not any(getattr(Board, direction)() for direction in ['up', 'down', 'left', 'right']):
                print("\nGame Over!")
                print(f"Final Score: {Board.score}")
                print(f"Total Moves: {moves_count}")
                break

            print(f"AI chose: {move}") 
            # Small delay to make the game visible
            time.sleep(0.5)
            moves_count += 1
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
    print("done")

create_dataset(4,10)