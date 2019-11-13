""" from https://github.com/keithito/tacotron """
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def phone_to_sequence(text):
  return [_symbol_to_id[s] for s in text.split()]
