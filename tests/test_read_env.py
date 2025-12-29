from dotenv import load_dotenv
import os

load_dotenv()

print("DB_HOST =", repr(os.getenv("DB_HOST")))
print("DB_USER =", repr(os.getenv("DB_USER")))
print("DB_PASSWORD =", repr(os.getenv("DB_PASSWORD")))
