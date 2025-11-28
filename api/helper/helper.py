import random

def generate_slug():
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz1234567890') for _ in range(10))