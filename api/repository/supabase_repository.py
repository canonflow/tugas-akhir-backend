# supabase_manager.py
import os
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """FastAPI dependency for Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    print("Supabase URL:", url)
    print("Supabase Key:", key)

    if not url or not key:
        raise EnvironmentError("Supabase credentials not found")
    
    return create_client(url, key)