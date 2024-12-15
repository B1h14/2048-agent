from openai import OpenAI
import Game_logic
import time
import re

THINKING_START_COUNT = 60

def get_ai_completion(client, messages, stream=False):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=stream
    )
    
    if not stream:
        return completion.choices[0].message.content
    
    # For streaming, return the stream object directly
    return completion

def get_chatgpt_action(client,Board,Board_size):
    messages = [
        {"role": "system", "content": "You are an AI playing 2048. Analyze the board and choose the best move. keep the biggest number in the corner always and try not to have small numbers trapped"},
        {"role": "user", "content": f"""Current board state:\n{str(Board)}\nChoose an action between: {Board.get_possible_actions()}\nAnswer with only one word in lowercase and make sure it is a valid action."""}
    ]
    move = get_ai_completion(client, messages).strip().lower()
    return move


def get_chatgpt_chain_of_thought_action(client,Board,Board_size):
    max_retries = 3
    possible_actions = Board.get_possible_actions()
    
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": "You are an AI playing 2048. Your goal is to analyze the board carefully and make the best move using step-by-step reasoning. Keep the largest numbers in the corner and avoid trapping small numbers."},
                {"role": "user", "content": f"""Current board state:\n{str(Board)}

Please analyze the board and choose a move following these steps:
1. Current Board Analysis:
   - Identify the location of the largest numbers
   - Note any potential merges
   - Identify any trapped small numbers

2. Available Moves: {possible_actions}
   For each possible move, evaluate:
   - How it affects number positions
   - Potential merges it creates
   - Risk of trapping small numbers
   - How it maintains corner strategy

3. Final Decision:
   Based on the analysis above, choose the best move.

Conclude your response with "CHOSEN_ACTION: <move>" where <move> is one of {possible_actions} in lowercase."""}
            ]
            print("Analyzing board state...")
            
            # Get streaming response
            stream = get_ai_completion(client, messages, stream=True)
            full_response = ""
            
            # Process the stream
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end='', flush=True)
                    full_response += content
            
            print("\n")  # Add newline after streaming
            response = full_response.strip().lower()
            
            # Extract the final action from the response
            move = None
            pattern = r"(?:CHOSEN_ACTION|Move):\s*(up|down|left|right)"

            # Search for the pattern in the response
            matches = re.findall(pattern, response, re.IGNORECASE)
                
            if matches:
                move = matches[-1].lower()  # Convert the action to lowercase for consistency
            else:
                break
            
            
            # Fallback: take the last word if format wasn't followed
            if not move:
                move = response.split()[-1]
            
            # Verify the move is valid
            if move in possible_actions:
                return move
            
            # If we get here, the move wasn't valid
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(1)  # Add a small delay between retries
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(1)
                continue
            
    # If all retries failed, return the first valid action as fallback
    if len(possible_actions) == 0:
        Board.end_game()
        return "no valid moves available"
    return possible_actions[0]


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
            if moves_count < THINKING_START_COUNT:
                move = get_chatgpt_action(client,Board,Board_Size)
            else:
                move = get_chatgpt_chain_of_thought_action(client,Board,Board_Size)
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
            if moves_count < THINKING_START_COUNT:
                time.sleep(0.1)
            else:
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

create_dataset(4,1)