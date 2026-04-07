"""User model and authentication utilities."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from passlib.context import CryptContext
from jose import JWTError, jwt

from .. import config

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "your-secret-key-change-in-production"  # TODO: Move to env
ALGORITHM = "HS256"


@dataclass
class User:
    user_id: str
    username: str
    email: str
    hashed_password: str
    created_at: str
    is_subscribed: bool = False
    free_runs_remaining: int = 5
    total_runs: int = 0
    otp_code: Optional[str] = None
    otp_expires_at: Optional[str] = None

    def verify_password(self, password: str) -> bool:
        # Bcrypt has a 72-byte limit, so truncate if necessary
        # Passlib expects a string, but bcrypt internally has 72-byte limit
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to 72 bytes, handling UTF-8 encoding properly
            truncated_bytes = password_bytes[:72]
            # Decode back to string (safely handle UTF-8 boundaries)
            while truncated_bytes:
                try:
                    password = truncated_bytes.decode('utf-8')
                    break
                except UnicodeDecodeError:
                    truncated_bytes = truncated_bytes[:-1]
            else:
                password = ""  # Fallback if all bytes invalid
        return pwd_context.verify(password, self.hashed_password)

    @staticmethod
    def hash_password(password: str) -> str:
        # Bcrypt has a 72-byte limit, so truncate if necessary
        # Passlib expects a string, but bcrypt internally has 72-byte limit
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to 72 bytes, handling UTF-8 encoding properly
            truncated_bytes = password_bytes[:72]
            # Decode back to string (safely handle UTF-8 boundaries)
            while truncated_bytes:
                try:
                    password = truncated_bytes.decode('utf-8')
                    break
                except UnicodeDecodeError:
                    truncated_bytes = truncated_bytes[:-1]
            else:
                password = ""  # Fallback if all bytes invalid
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict) -> str:
        to_encode = data.copy()
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None

