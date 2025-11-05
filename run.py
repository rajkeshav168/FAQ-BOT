from app import app

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, port=5000)