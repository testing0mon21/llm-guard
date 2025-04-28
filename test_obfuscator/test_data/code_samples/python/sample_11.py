def verify_jwt_token(token, secret_key):
    """Verify a JWT token."""
    import jwt
    
    try:
        decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
        return True, decoded
    except jwt.InvalidTokenError:
        return False, None