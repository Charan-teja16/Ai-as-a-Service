"""User management service."""
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .. import config
from ..models.user import User
from ..utils.state import JSONStore


class UserService:
    def __init__(self):
        config.ensure_directories()
        self._store = JSONStore(config.STORAGE_DIR / "users_index.json")

    def create_user(self, username: str, email: str, password: str) -> User:
        """Create a new user."""
        # Check if email already exists
        for user_data in self._store.list():
            if user_data.get("email") == email:
                raise ValueError("Email already registered")
            if user_data.get("username") == username:
                raise ValueError("Username already taken")

        user_id = str(uuid.uuid4())
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            hashed_password=User.hash_password(password),
            created_at=datetime.utcnow().isoformat(),
            free_runs_remaining=5,
        )
        self._store.set(user_id, user.__dict__)
        return user

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user_data in self._store.list():
            if user_data.get("email") == email:
                return User(**user_data)
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        user_data = self._store.get(user_id)
        if not user_data:
            return None
        return User(**user_data)

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user and return user if valid."""
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not user.verify_password(password):
            return None
        return user

    def set_otp(self, email: str, otp_code: str) -> None:
        """Set OTP for password reset."""
        user = self.get_user_by_email(email)
        if not user:
            return
        user.otp_code = otp_code
        user.otp_expires_at = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
        self._store.set(user.user_id, user.__dict__)

    def verify_otp(self, email: str, otp_code: str) -> bool:
        """Verify OTP code."""
        user = self.get_user_by_email(email)
        if not user or not user.otp_code:
            return False
        if user.otp_code != otp_code:
            return False
        if user.otp_expires_at:
            expires = datetime.fromisoformat(user.otp_expires_at)
            if datetime.utcnow() > expires:
                return False
        return True

    def reset_password(self, email: str, new_password: str) -> None:
        """Reset user password."""
        user = self.get_user_by_email(email)
        if not user:
            raise ValueError("User not found")
        user.hashed_password = User.hash_password(new_password)
        user.otp_code = None
        user.otp_expires_at = None
        self._store.set(user.user_id, user.__dict__)

    def decrement_free_runs(self, user_id: str) -> bool:
        """Decrement free runs. Returns True if successful, False if no runs left."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        if user.is_subscribed:
            return True  # Subscribed users have unlimited
        if user.free_runs_remaining <= 0:
            return False
        user.free_runs_remaining -= 1
        user.total_runs += 1
        self._store.set(user_id, user.__dict__)
        return True

    def subscribe_user(self, user_id: str) -> None:
        """Subscribe user (unlimited training)."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        user.is_subscribed = True
        self._store.set(user_id, user.__dict__)

    def update_user(self, user: User) -> None:
        """Update user data."""
        self._store.set(user.user_id, user.__dict__)



