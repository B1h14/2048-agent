import pygame 
import Layout
import Game_logic
import sys
import Brain
import AI_integration


Board = Game_logic.Board()
Model = Brain.CNN()


# Initialize Pygame
pygame.init()

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
                AI_integration.use_ai(Board,Board.size,Model)
    screen.fill(Layout.white)
    Layout.draw_Board(Board.B,Board.size,screen,font,Board.score )
    pygame.display.flip()

pygame.quit()
sys.exit()




