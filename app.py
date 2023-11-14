from flask import Flask, render_template, request, redirect, url_for
from cbfs import cbfs
from werkzeug.utils import secure_filename
import os
from pydub import AudioSegment
import uuid
from gtts import gTTS
import whisper
import base64
from flask import jsonify

app = Flask(__name__)

# Sample data for replies (you can replace this with your chatbot logic)
replies = {}
c = cbfs()
audio_model = whisper.load_model("base")

@app.route('/')
def index():
    return render_template('index.html', replies=replies)

@app.route('/submit_question', methods=['POST'])
def submit_question():
    question = request.form.get('question')
    reply = c.convchain(question)
    replies[question] = reply
    return index()

@app.route('/start_new_chat', methods=['POST'])
def start_new_chat():
    # Clear the conversation history to start a new chat
    c.clr_history()
    replies.clear()
    return redirect(url_for('index'))

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return 'No audio file', 400

    audio = request.files['audio']
    if audio.filename == '':
        return 'No selected file', 400

    if audio:

        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join('audios', filename)

        original_audio = AudioSegment.from_file(audio)
        original_audio.export(filepath, format="mp3")

        result = audio_model.transcribe(filepath, language="en")

        question = result["text"]

        print("The question is: "+ question)
              

        reply = c.convchain(question)  # Replace with your chatbot logic
        replies[question] = reply

        print("The reply is: "+ reply)

        reply_filename = f"{uuid.uuid4()}.mp3"
        reply_filepath = os.path.join('audios', reply_filename)
        mp3_obj = gTTS(text=reply, lang="en", slow=False)
        mp3_obj.save(reply_filepath)

        with open(reply_filepath, "rb") as mp3_file:
            encoded_mp3 = base64.b64encode(mp3_file.read()).decode('utf-8')

        # mp3_obj = gTTS(text=reply, lang="en", slow=False)

        return jsonify({
            'replies': replies,
            'encoded_mp3': f"data:audio/mpeg;base64,{encoded_mp3}"
        })

if __name__ == '__main__':
    app.run(debug=True)
