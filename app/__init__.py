from flask import Flask

print("Initializing Flask application...")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

print("Loading routes...")
from app import routes

print("Application initialized successfully!")