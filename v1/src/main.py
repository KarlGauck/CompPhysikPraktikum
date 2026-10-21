from rendering import Window
from nbody import NBodySystem

import numpy as np

    
def render():
    pass    


if __name__ == "__main__":
    window = Window(1920, 1080, "N Body")

    stop = False

    system = NBodySystem(
        [np.array([0, 0]), np.array([1, 1])],
        [1, 1]
    ) 

    system.step(1)
    

    while not stop:
        window.circle(50, 50, 100, (0, 0, 0))
        window.update()

        if window.should_close():
            stop = True
