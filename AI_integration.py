import Brain
import torch
import numpy 
import Game_logic
import random
import pygame
import Layout
import sys

def convert_to_input(Board_list,Board_size):
    L = [Board_list[i//Board_size][ i%Board_size]    for i in range(Board_size**2) ]
    Inp = torch.tensor(L,dtype=torch. float32)
    return Inp
def sort_args(L):
    Li = [(L[i] , i) for i in range(len(L))]
    Li.sort(key=lambda x:x[0],reverse= True)
    return [ Li[i][1] for i in range(len(L)) ]
def determine_Action(Board,Model_output):
    Args_sorted = sort_args(Model_output)
    k = 0
    while not ( perform_Action(k,Board)):
        k+=1
        if k == 4:
            Board.end_game()
            break
def perform_Action(i , Board):
    if i == 0:
        return Board.down()
    elif i ==1 :
        return Board.up()
    elif i == 2:
        return Board.left()
    elif i == 3:
        return Board.right()
    else :
        return True
def use_ai(Board,Board_size,Model):
    Inp = convert_to_input(Board.B,Board_size)
    Out = Model.forward(Inp)
    determine_Action(Board,Out)

# Initialize genetic algorithm parameters
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        model = Brain.CNN()
        population.append(model)
    return population
# Crossover operator: Single-point crossover
def crossover(parent1, parent2):
    child = Brain.CNN()
    with torch.no_grad():
        for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
            mask = torch.rand_like(p1) > 0.5
            c.data = torch.where(mask, p1.data, p2.data)  # Combine weights
    return child
 #Mutation operator: Random mutation
def mutate(model , mutation_rate):
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            param.data += torch.randn_like(param.data) * 0.1  # Adding Gaussian noise with std=0.1
    return model
def run_model(model):
    Board = Game_logic.Board()
    while not(Board.game_over):
        use_ai(Board,4,model)
    return Board.score + max_in_board(Board.B)**2
def max_in_board(Board_list):
    m = 0
    for i in range(len(Board_list)):
        for j in range(len(Board_list)):
            if Board_list[i][j]>m:
                m=Board_list[i][j]
    return m
def run_gereneration(population,population_size,num_games,mutation_rate):
    Performances = []
    for model in population:
        scores = []
        for i in range(num_games):
            scores.append(run_model(model)) 
        performance = sum(scores)/num_games


        Performances.append([model,performance])
    Performances.sort(key=lambda x: x[1], reverse=True)
    print("the best performance is :" , Performances[0][1])
    top_individuals = Performances[:len(population) // 2]
    next_generation = [ top_individuals[i][0] for i in range(len(population) //5)]
    while len(next_generation) < population_size:
        parents = random.sample(top_individuals, 2)  # Select 2 parents
        child = crossover(parents[0][0], parents[1][0])  # Combine genes
        child = mutate(child, mutation_rate)  # Mutate
        next_generation.append(child)
    
    population = next_generation
def train_loop_genetic_algorithm(population_size,mutation_rate,num_generations,number_of_games):
    population = initialize_population(population_size)
    for i in range(num_generations):
        print("generation :" , i+1)
        run_gereneration(population,population_size,number_of_games,mutation_rate)
    return population
def Display_ai(model):
    pygame.init()
    Board = Game_logic.Board()
    # Set the dimensions of the window
    width, height = 800, 600

    # Create the window
    screen = pygame.display.set_mode((width, height))

    # Set the window title
    pygame.display.set_caption("2048")


    # Initialize font
    font = pygame.font.Font(None, 36)

    # Main loop

    running = True
    while running:
        #Layout.define_key_actions(Board,pygame.event)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Handle window close
                running = False
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_SPACE:
                    use_ai(Board,Board.size,model)
                if event.key == pygame.K_r:
                    Board = Game_logic.Board()
        screen.fill(Layout.white)
        Layout.draw_Board(Board.B,Board.size,screen,font,Board.score )
        pygame.display.flip()

    pygame.quit()
    sys.exit()
population =train_loop_genetic_algorithm(40,0.3,10000,50)

torch.save(population[0].state_dict(), "model.pth")

Display_ai(population[0])



