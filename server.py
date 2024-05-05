from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, shortuuid, datetime, shutil, pickle

app = Flask(__name__)
CORS(app)  # Allow requests from all origins

UPLOAD_FOLDER = 'clients'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def findClientDir(token):
    for client in os.listdir(app.config['UPLOAD_FOLDER']):
        if (token == client.split(".")[1]):
            return os.path.join(app.config['UPLOAD_FOLDER'], client)
    return ""

#--------------------------------------------------------------------#

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getToken')
def generateToken():
    msg = request.args.get('message')
    token = ""

    #delete expired sessions
    for client in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], client)):
            if (datetime.datetime.now() - datetime.datetime.strptime(client.split('.')[0],'%Y-%m-%d_%H-%M-%S') > datetime.timedelta(minutes=1)):
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

@app.route('/save', methods=['POST'])
def saveSession():
    clientDir = findClientDir(request.args.get('token'))
    data = request.json
    
    # Serialize the object into bytes
    binary_data = pickle.dumps(data)

    # Write the binary data to a file
    with open(os.path.join(clientDir,'log.bin'), 'wb') as file:
        file.write(binary_data)

    return jsonify({'message': 'Session saved successfully'})


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

    print(msg)

    # Return decoded data as JSON response
    return jsonify({'session': msg})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)