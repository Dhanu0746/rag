from src.database import Base, engine
from src.models import User

print("Creating database...")

Base.metadata.create_all(bind=engine)

print("Database created successfully!")