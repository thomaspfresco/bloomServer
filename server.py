from flask import Flask, request, jsonify, render_template, send_file, make_response
from flask_cors import CORS
import os, shortuuid, datetime, shutil, pickle, json
import numpy as np
import multiprocessing as mp
import io
#import threading
#import queue


from musicGeneration import generateMelody, generateHarmony, generateBass, generateDrums

#from basic_pitch.inference import predict
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

@app.route('/getToken')
def generateToken():
    msg = request.args.get('message')
    token = ""

    #delete expired sessions
    for client in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], client)):
            if (datetime.datetime.now() - datetime.datetime.strptime(client.split('.')[0],'%Y-%m-%d_%H-%M-%S') > datetime.timedelta(days=1)):
                shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], client))
                print("session deleted: "+client)
    
    #user already has a token
    if (msg != "New user"):
        for client in os.listdir(app.config['UPLOAD_FOLDER']):
            #user has a valid token
            if (msg == client.split(".")[1]):
                token = msg
                try:
                    os.rename(os.path.join(app.config['UPLOAD_FOLDER'],client), os.path.join(app.config['UPLOAD_FOLDER'],datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.'+token))
                    print("Expire time reset with success for "+token)
                except OSError as e:
                    print("Error: {e}")
                return jsonify({"message":"Your token still valid", "token": token})
     
    token = str(shortuuid.uuid())
    
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'],datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.'+token))
    
    #user don't have token
    if (msg == "New user"):
        return jsonify({"message":"New user", "token": token})
    #user has an expired token, reset session
    else:
        return jsonify({"message":"Your session expired","token": token})

#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

@app.route('/upload', methods=['POST'])
def uploadFiles():
    token = request.args.get('token')
    uploaded_files = request.files.getlist('files')
    filePaths = []

    #find the directory of the client
    clientDir = findClientDir(token)

    for file in uploaded_files:
        file_path = os.path.join(clientDir, file.filename)
        file.save(file_path)
        filePaths.append(file_path)
    return jsonify({'message': 'Files uploaded successfully', 'file_paths': filePaths})

#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

@app.route('/save', methods=['POST'])
def saveSession():
    clientDir = findClientDir(request.args.get('token'))
    data = request.json

    binary_data = pickle.dumps(data)

    # Write the binary data to a file
    with open(os.path.join(clientDir,'log.bin'), 'wb') as file:
        file.write(binary_data)

    return jsonify({'message': 'Session saved successfully'})

#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

@app.route('/load', methods=['GET'])
def loadSession():
    clientDir = findClientDir(request.args.get('token'))

    if (os.path.exists(os.path.join(clientDir,'log.bin'))):
    
        # Read binary data from a file
        with open(os.path.join(clientDir,'log.bin'), 'rb') as file:
            binary_data = file.read()

        # Decode binary data using pickle
        msg = pickle.loads(binary_data)
    else:
        msg = "bin file not found"

    #print(msg)

    # Return decoded data as JSON response
    return jsonify({'session': msg})

#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

'''@app.route('/basicpitch', methods=['POST'])
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

        return 'No WAV file received.', 400'''
    
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

@app.route('/generate', methods=['POST'])
def generate():
    clientDir = findClientDir(request.args.get('token'))
    
    if (clientDir != ""):

        data = request.json
        
        trackName = request.args.get('trackName')
        parts = {} 

        if trackName == "DRUMS":
            processesResults = mp.Queue()

            processes = [mp.Process(target=generateDrums, args=(data, processesResults)),
                mp.Process(target=generateDrums, args=(data, processesResults)),
                mp.Process(target=generateDrums, args=(data, processesResults)),
                mp.Process(target=generateDrums, args=(data, processesResults))]

            for p in processes: p.start()
            for p in processes: p.join()

            index = 0
            while not processesResults.empty():
                parts[index] = processesResults.get()
                index += 1

        else:
            #generate 4 parts
            for i in range(4):

                if trackName == "MELODY":
                    parts[i] = generateMelody(data)
                
                elif trackName == "HARMONY":

                    if i <= 1:
                        neutral = False
                    else:
                        neutralProb = np.random.rand()
                        if neutralProb > 0.8: neutral = True
                        else: neutral = False

                    parts[i] = generateHarmony(data, neutral)
                
                elif trackName == "BASS":
                    parts[i] = generateBass(data)      
    
        #print(parts)
        return json.dumps(parts, indent=4)

@app.route('/renderMidi', methods=['POST'])
def renderMidi():
    clientDir = findClientDir(request.args.get('token'))
    
    if (clientDir != ""):

        data = request.json
        tempo = int(request.args.get('tempo'))
        timeBtwSteps = (60 / tempo / 4)

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        #midi.add_tempo_change(tempo=tempo, time=0)

        for trackName in data:
            filename = trackName + '.mid'
            for n in data[trackName]:
                note = data[trackName][n]
                if (trackName == "DRUMS"):
                    #ajust to DAWs standard drum kit
                    if note['pitch'] == 1: note['pitch'] = 2 # snare
                    elif note['pitch'] == 2: note['pitch'] = 6 # closed hi-hat
                    elif note['pitch'] == 3: note['pitch'] = 10 # open hi-hat
                    elif note['pitch'] == 4: note['pitch'] = 9 # high tom
                    elif note['pitch'] == 5: note['pitch'] = 7 # low tom
                    elif note['pitch'] == 6: note['pitch'] = 13 # crash cymbal
                    midiNote = pretty_midi.Note(velocity=127, pitch=note['pitch']+12*3, start=note['start']*timeBtwSteps, end=(note['start']+note['duration'])*timeBtwSteps)
                else:
                    midiNote = pretty_midi.Note(velocity=127, pitch=note['pitch']+12*note['octave'], start=note['start']*timeBtwSteps, end=(note['start']+note['duration'])*timeBtwSteps)

                
                instrument.notes.append(midiNote)


        midi.instruments.append(instrument)

        # Write the MIDI data to a BytesIO object
        midi_data = io.BytesIO()
        midi.write(midi_data)
        midi_data.seek(0)  # Rewind the buffer

        # Prepare the response with the MIDI file
        response = make_response(send_file(
            midi_data,
            as_attachment=True,
            download_name=filename,
            mimetype='audio/midi'
        ))

        # Set a custom header for the filename
        response.headers['filename'] = filename
        response.headers['Access-Control-Expose-Headers'] = 'filename'
        return response
        
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)