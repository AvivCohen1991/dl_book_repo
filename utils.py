class Log:
    def __init__(self, file_name):
        self.counter = 0
        self.file_name = file_name
        self.clean()

    def print(self, msgs):
        with open(self.file_name, 'a') as file:
            file.write('*****' + str(self.counter) + '*******\n')
            for msg in msgs:
                file.write(str(msg))
            self.counter += 1.

    def clean(self):
        with open(self.file_name, 'w') as file:
            file.write('\n')
