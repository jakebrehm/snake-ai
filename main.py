import pygame

from snake_ai import game


if __name__ == '__main__':

    game = game.SnakeGame()

    # Start the game loop
    while True:
        # Advance the game
        game_over, score = game.play_step()

        # Break if game over
        if game_over:
            break
    
    print(f"Final score: {score}")
    
    pygame.quit()