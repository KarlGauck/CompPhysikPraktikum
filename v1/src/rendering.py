import pygame

class Window:

    def __init__(self, width, height, title):
        self.width = width
        self.height = height
        self.title = title

        self.window = pygame. display.set_mode((width, height))
        pygame.display.set_caption(title)

    def should_close(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def update(self):
        pygame.display.update()
        self.window.fill((255, 255, 255))

    def circle(self, x, y, r, color):
        pygame.draw.circle(self.window, color, [x, y], r, 0) 
