def generate_password(length=12):
    """Generate a random password with the specified length."""
    import random
    import string
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(random.choice(characters) for _ in range(length))
    return password