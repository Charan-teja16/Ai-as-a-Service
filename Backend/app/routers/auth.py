"""Authentication routes."""
import random
import string
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..schemas import (
    RegisterRequest,
    LoginRequest,
    ForgotPasswordRequest,
    VerifyOTPRequest,
    ResetPasswordRequest,
    AuthResponse,
    UserInfoResponse,
    SubscribeRequest,
)
from ..services.user_service import UserService
from ..services.email_service import EmailService
from ..models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()
user_service = UserService()
email_service = EmailService()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    payload = User.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register a new user."""
    if request.password != request.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    try:
        user = user_service.create_user(request.username, request.email, request.password)
        token = User.create_access_token({"sub": user.user_id})
        return AuthResponse(
            access_token=token,
            user={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "is_subscribed": user.is_subscribed,
                "free_runs_remaining": user.free_runs_remaining,
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login user."""
    user = user_service.authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = User.create_access_token({"sub": user.user_id})
    return AuthResponse(
        access_token=token,
        user={
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "is_subscribed": user.is_subscribed,
            "free_runs_remaining": user.free_runs_remaining,
        }
    )


@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Send OTP for password reset."""
    user = user_service.get_user_by_email(request.email)
    if not user:
        # Don't reveal if email exists for security
        return {"message": "If email exists, OTP has been sent"}
    
    otp_code = "".join(random.choices(string.digits, k=6))
    user_service.set_otp(request.email, otp_code)
    email_service.send_otp(request.email, otp_code)
    return {"message": "OTP sent to email"}


@router.post("/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """Verify OTP code."""
    is_valid = user_service.verify_otp(request.email, request.otp_code)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    return {"message": "OTP verified successfully"}


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset password after OTP verification."""
    if request.new_password != request.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(request.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    if not user_service.verify_otp(request.email, request.otp_code):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    user_service.reset_password(request.email, request.new_password)
    return {"message": "Password reset successfully"}


@router.get("/me", response_model=UserInfoResponse)
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserInfoResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        is_subscribed=user.is_subscribed,
        free_runs_remaining=user.free_runs_remaining,
        total_runs=user.total_runs,
    )


@router.post("/subscribe")
async def subscribe(request: SubscribeRequest, user: User = Depends(get_current_user)):
    """Subscribe user for unlimited training."""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Subscription confirmation required")
    
    user_service.subscribe_user(user.user_id)
    return {"message": "Successfully subscribed! You now have unlimited training."}



