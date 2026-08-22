import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

VALID_API_KEY = os.getenv("PAPERMIND_API_KEY")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")