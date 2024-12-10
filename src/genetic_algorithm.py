from typing import List, Optional
import random
import sys

import numpy as np
import pygame
import torch

import Brain
import Game_logic
import Layout


def initialize_population(population_size: int) -> List[Brain.NN]:
    """Initialize a population of neural networks.
    
    Args:
        population_size: Number of models to create
        
    Returns:
        List of NN models
    """
    return [Brain.NN() for _ in range(population_size)]


def crossover(parent1: Brain.NN, parent2: Brain.NN) -> Brain.NN:
    """Perform single-point crossover between two parent models.
    
    Args:
        parent1: First parent model
        parent2: Second parent model
        
    Returns:
        Child model with combined weights
    """
    child = Brain.NN()
    with torch.no_grad():
        for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
            mask = torch.rand_like(p1) > 0.5
            c.data = torch.where(mask, p1.data, p2.data)
    return child


def mutate(model: Brain.NN, mutation_rate: float) -> Brain.NN:
    """Apply random mutations to model parameters.
    
    Args:
        model: Model to mutate
        mutation_rate: Probability of mutation for each parameter
        
    Returns:
        Mutated model
    """
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            param.data += torch.randn_like(param.data) * 0.1  # Adding Gaussian noise with std=0.1
    return model


def run_model(model: Brain.NN) -> float:
    """Run a single game using the given model.
    
    Args:
        model: Model to evaluate
        
    Returns:
        Game score plus square of maximum tile value
    """
    board = Game_logic.Board()
    while not board.game_over:
        use_ai(board, 4, model)
    return board.score + max_in_board(board.B)**2


def max_in_board(board_list: List[List[int]]) -> int:
    """Find the maximum value in the game board.
    
    Args:
        board_list: 2D list representing the game board
        
    Returns:
        Maximum value found
    """
    return max(max(row) for row in board_list)


def run_generation(population: List[Brain.NN], population_size: int, 
                  num_games: int, mutation_rate: float) -> List[Brain.NN]:
    """Run one generation of the genetic algorithm.
    
    Args:
        population: Current population of models
        population_size: Target size for next generation
        num_games: Number of games to evaluate each model
        mutation_rate: Probability of mutation
        
    Returns:
        Next generation of models
    """
    performances = []
    for model in population:
        scores = [run_model(model) for _ in range(num_games)]
        performance = sum(scores) / num_games
        performances.append([model, performance])
    
    performances.sort(key=lambda x: x[1], reverse=True)
    print(f"Best performance: {performances[0][1]}")
    
    top_individuals = performances[:len(population) // 2]
    next_generation = [model for model, _ in top_individuals[:len(population) // 5]]
    
    while len(next_generation) < population_size:
        parent1, parent2 = random.sample(top_individuals, 2)
        child = crossover(parent1[0], parent2[0])
        child = mutate(child, mutation_rate)
        next_generation.append(child)
    
    return next_generation


def train_genetic_algorithm(population_size: int, mutation_rate: float,
                          num_generations: int, number_of_games: int) -> List[Brain.NN]:
    """Train models using genetic algorithm.
    
    Args:
        population_size: Number of models in each generation
        mutation_rate: Probability of mutation
        num_generations: Number of generations to train
        number_of_games: Games per model for evaluation
        
    Returns:
        Final population of trained models
    """
    population = initialize_population(population_size)
    for gen in range(num_generations):
        print(f"Generation: {gen + 1}")
        population = run_generation(population, population_size, number_of_games, mutation_rate)
    return population


def display_ai(model: Brain.NN) -> None:
    """Display AI gameplay using pygame.
    
    Args:
        model: Model to visualize
    """
    pygame.init()
    board = Game_logic.Board()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("2048")
    font = pygame.font.Font(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    use_ai(board, board.size, model)
                if event.key == pygame.K_r:
                    board = Game_logic.Board()
        
        screen.fill(Layout.white)
        Layout.draw_Board(board.B, board.size, screen, font, board.score)
        pygame.display.flip()

    pygame.quit()
    sys.exit()
