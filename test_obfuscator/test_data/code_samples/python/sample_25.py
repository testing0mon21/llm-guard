def connect_to_postgres(db_params):
    """Connect to a PostgreSQL database."""
    import psycopg2
    
    conn = None
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        if conn is not None:
            conn.close()
        return None