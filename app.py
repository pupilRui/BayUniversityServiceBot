from flask import Flask, render_template, request, redirect, url_for
from cbfs import cbfs

app = Flask(__name__)

# Sample data for replies (you can replace this with your chatbot logic)
replies = {}
c = cbfs()

@app.route('/')
def index():
    return render_template('index.html', replies=replies)

@app.route('/submit_question', methods=['POST'])
def submit_question():
    question = request.form.get('question')
    reply = c.convchain(question)  # Replace with your chatbot logic
    replies[question] = reply
    return index()

@app.route('/start_new_chat', methods=['POST'])
def start_new_chat():
    # Clear the conversation history to start a new chat
    c.clr_history()
    replies.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
