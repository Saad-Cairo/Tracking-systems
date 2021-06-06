

class ID_generator:

    def __init__(self):
        self.current_ID = -1

    def assign_ID(self):
        self.current_ID += 1
        return self.current_ID