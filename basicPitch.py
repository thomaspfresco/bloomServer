from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, shutil, pickle, json
import numpy as np

from basic_pitch.inference import predict
import pretty_midi

#--------------------------------------------------------------------#

app = Flask(__name__)
CORS(app) # Allow requests from all origins

@app.route('/')
def index():
    return render_template('index.html')

#--------------------------------------------------------------------#

UPLOAD_FOLDER = 'clients'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def findClientDir(token):
    for client in os.listdir(app.config['UPLOAD_FOLDER']):
        if (token == client.split(".")[1]):
            return os.path.join(app.config['UPLOAD_FOLDER'], client)
    return ""

#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

@app.route('/basicpitch', methods=['POST'])
def basicPitch():
    uploaded_file = request.files['file']  # 'file' should match the key used in FormData
    
    clientDir = findClientDir(request.args.get('token'))

    if (clientDir != ""):

        if uploaded_file:
            file_path = os.path.join(clientDir, uploaded_file.filename)
            uploaded_file.save(file_path)

            tempo = int(request.args.get('tempo'))
            timeBtwSteps = (60 / tempo / 4) * 1000

            model_output, midi_data, note_events = predict(file_path,
                                                    midi_tempo=tempo,
                                                    onset_threshold = 0.5,
                                                    frame_threshold = 0.3,
                                                    minimum_note_length=timeBtwSteps)

            dataGenerated = {}

            compensation = 2

            for instrument in midi_data.instruments:
                instrument.pitch_bends.clear()
                
                i = 0
                
                for note in instrument.notes:
                    noteInfo = {}
                    
                    step = round((note.start*1000) / timeBtwSteps)
                    
                    if step >= 2:
                        step -= compensation
                    
                    noteInfo['start'] = int(step)
                    duration = round((note.end*1000 - note.start*1000) / timeBtwSteps)
                    noteInfo['duration'] = int(duration)
                   
                    octave = 0
                    pitch = note.pitch

                    while (pitch >= 12):
                        pitch -= 12
                        octave += 1
                    noteInfo['octave'] = int(octave)
                    noteInfo['pitch'] = int(pitch)

                    dataGenerated[int(i)] = noteInfo
                    
                    i += 1

            #delete audio file
            os.remove(file_path)

            return jsonify(dataGenerated)

        return 'No WAV file received.', 400
    
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)