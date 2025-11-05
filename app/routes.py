from flask import render_template, request, jsonify
from app import app
from app.bot import JupiterFAQBot

bot = JupiterFAQBot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    
    response = bot.generate_response(question)
    suggestions = bot.suggest_related_queries(question)
    
    return jsonify({
        'response': response,
        'suggestions': suggestions
    })