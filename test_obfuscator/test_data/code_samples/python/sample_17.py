def hash_password(password, salt=None):
    """Hash a password using PBKDF2."""
    import hashlib
    import os
    
    if salt is None:
        salt = os.urandom(32)
    
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return {'salt': salt, 'key': key}