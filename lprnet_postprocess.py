from utils import decode_function, BeamDecoder

CHARS = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
     'Y', 'X', '-'
]

def lprnet_decode(predictions):
    return decode_function(predictions, CHARS, BeamDecoder)
