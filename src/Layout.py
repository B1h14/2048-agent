import sys
import Game_logic
import pygame


white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 128, 255)

# Table dimensions
table_rows = 4
table_cols = 4
cell_width = 100  # Width of each cell
cell_height = 100  # Height of each cell
table_x = 200  # Top-left x-coordinate of the table
table_y = 100  # Top-left y-coordinate of the table

def draw_Board(Board_list,Board_size,screen,font,score):
    for row in range(Board_size):
        for col in range(Board_size):
            # Calculate cell position
            cell_x = table_x + col * cell_width
            cell_y = table_y + row * cell_height

            # Draw cell rectangle (optional visual aid)
            pygame.draw.rect(screen, blue, (cell_x, cell_y, cell_width, cell_height), 2)

            # Add text to the cell
            cell_text = str(Board_list[row][col])
            text_surface = font.render(cell_text, True, black)
            text_rect = text_surface.get_rect(center=(cell_x + cell_width / 2, cell_y + cell_height / 2))
            screen.blit(text_surface, text_rect)
    text_surface = font.render("score =" + str(score), True, black)
    text_rect = text_surface.get_rect(center=( 100,100))
    screen.blit(text_surface, text_rect)
def define_key_actions(Board,Event,Model):
    for event in Event.get():
        if event.type == pygame.QUIT:  # Handle window close
            running = False
        if event.type == pygame.KEYDOWN: 
            if event.key == pygame.K_UP:
                Board.up()
            elif event.key == pygame.K_DOWN:
                Board.down()
            elif event.key == pygame.K_LEFT:
                Board.left()
            elif event.key == pygame.K_RIGHT:
                Board.right()

