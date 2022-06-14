class WrongMethodError(Exception):
    def __init__(self, message='WrongMethodError: 해당 HTTP 메소드가 없습니다.'):
        self.message = message

    def __str__(self):
        print('calling str')
        return '[MyCustomError], {0} '.format(self.message)


class NoneArgumentError(Exception):
    def __init__(self, message='NoneArgumentError: 해당 인자가 없습니다.'):
        self.message = message

    def __str__(self):
        print('calling str')
        return '[MyCustomError], {0} '.format(self.message)
