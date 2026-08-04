from datetime import datetime, timedelta
import os

from jose import JWTError, jwt
# pyrefly: ignore [missing-import]
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.database import SessionLocal
from src.models import User

# ==========================
# Password Hashing
# ==========================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "1"))



# ==========================
# Password Functions
# ==========================

def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str):
    return pwd_context.verify(password, hashed_password)


# ==========================
# JWT Functions
# ==========================

def create_token(username: str):
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str):
    try:
        return jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
    except JWTError:
        return None


# ==========================
# Register User
# ==========================

def register_user(username: str, password: str):

    db: Session = SessionLocal()

    try:

        existing = (
            db.query(User)
            .filter(User.username == username)
            .first()
        )

        if existing:
            return False, "Username already exists."

        user = User(
            username=username,
            password=hash_password(password),
        )

        db.add(user)
        db.commit()

        return True, "Registration successful."

    finally:
        db.close()


# ==========================
# Login User
# ==========================

def login_user(username: str, password: str):

    db: Session = SessionLocal()

    try:

        user = (
            db.query(User)
            .filter(User.username == username)
            .first()
        )

        if user is None:
            return False, "User not found."

        if not verify_password(password, user.password):
            return False, "Incorrect password."

        token = create_token(username)

        return True, token

    finally:
        db.close()