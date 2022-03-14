class controller_timer:

    def __init__(self):
        pass

    def __str__(self):
        return "timer"

    def run_step(self, control_step: int):
        if control_step % 4 == 0:
            return [1, 1, 0, 0, 0, 0, 0, 0]
        if control_step % 4 == 1:
            return [0, 0, 1, 1, 0, 0, 0, 0]
        if control_step % 4 == 2:
            return [0, 0, 0, 0, 1, 1, 0, 0]
        if control_step % 4 == 3:
            return [0, 0, 0, 0, 0, 0, 1, 1]
