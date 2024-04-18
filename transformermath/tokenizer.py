class Tokenizer:
    def __init__(self, replacer):
        self.REPLACER = replacer

class AdditionTokenizer(Tokenizer):
    def __init__(self):
        super().__init__({
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '+': 10,
            '=': 11, 'E': 12, 'P': 13
        })

class SubtractionTokenizer(Tokenizer):
    def __init__(self):
        super().__init__({
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '-': 10,
            '=': 11, 'E': 12, 'P': 13
        })

class SequencesTokenizer(Tokenizer):
    def __init__(self):
        super().__init__({
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<': 10,
            '>': 11, 'E': 12, 'P': 13
        })

class MultiplicationTokenizer(Tokenizer):
    def __init__(self):
        super().__init__({
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '*': 10,
            '=': 11, 'E': 12, 'P': 13
        })

class MixedTokenizer(Tokenizer):
    def __init__(self):
        super().__init__({
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '+': 10,
            '=': 11, 'E': 12, 'P': 13, '-':14, '<':15, '>':16
        })
