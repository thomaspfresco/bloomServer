import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models once
harmonyBassModel = load_model('models/harmonyAndBass.h5')

drumModels = {
    1: load_model('models/drums1.h5'),
    2: load_model('models/drums2.h5'),
    3: load_model('models/drums3.h5'),
    4: load_model('models/drums4.h5'),
    'brake': load_model('models/brakes.h5')
}

durationModels = {
    2: load_model('models/durations2.h5'),
    3: load_model('models/durations3.h5'),
    4: load_model('models/durations4.h5'),
    5: load_model('models/durations5.h5')
}

#signatures (notes/roots)
keys = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}

# major scales
majorScales = {
    0: [0, 2, 4, 5, 7, 9, 11],
    1: [1, 3, 5, 6, 8, 10, 0],
    2: [2, 4, 6, 7, 9, 11, 1],
    3: [3, 5, 7, 8, 10, 0, 2],
    4: [4, 6, 8, 9, 11, 1, 3],
    5: [5, 7, 9, 10, 0, 2, 4],
    6: [6, 8, 10, 11, 1, 3, 5],
    7: [7, 9, 11, 0, 2, 4, 6],
    8: [8, 10, 0, 1, 3, 5, 7],
    9: [9, 11, 1, 2, 4, 6, 8],
    10: [10, 0, 2, 3, 5, 7, 9],
    11: [11, 1, 3, 4, 6, 8, 10]
}

# 1:I, 2:II, 3:III, 4:IV, 5:V, 6:VI, 7:VII
majorChordsSet = {
    1: 'maj',
    2: 'min',
    3: 'min',
    4: 'maj',
    5: 'maj',
    6: 'min',
    7: 'dim'
}

# minor scales
minorScales = {
    0: [0, 2, 3, 5, 7, 8, 10],
    1: [1, 3, 4, 6, 8, 9, 11],
    2: [2, 4, 5, 7, 9, 10, 0],
    3: [3, 5, 6, 8, 10, 11, 1],
    4: [4, 6, 7, 9, 11, 0, 2],
    5: [5, 7, 8, 10, 0, 1, 3],
    6: [6, 8, 9, 11, 1, 2, 4],
    7: [7, 9, 10, 0, 2, 3, 5],
    8: [8, 10, 11, 1, 3, 4, 6],
    9: [9, 11, 0, 2, 4, 5, 7],
    10: [10, 0, 1, 3, 5, 6, 8],
    11: [11, 1, 2, 4, 6, 7, 9]
}

# 1:I, 2:II, 3:III, 4:IV, 5:V, 6:VI, 7:VII
minorChordsSet = {
    1: 'min',
    2: 'dim',
    3: 'maj',
    4: 'min',
    5: 'min',
    6: 'maj',
    7: 'maj'
}

# modifier
# 0: nothing -> maj/min/dim
# 1: add 7th -> maj7/min7
# 2: cut 3rd (root + 5th) -> neutral
def getChordNotes(key, chordDegree, modifier):
    notes = []

    chordType = majorChordsSet[chordDegree]

    rootIndex = majorScales[key][chordDegree-1]
    
    #major scale - root, (3rd), 5th, (7th)
    if chordType == 'maj':
        if (modifier == 1): notes = [majorScales[rootIndex][0], majorScales[rootIndex][2], majorScales[rootIndex][4], majorScales[rootIndex][6]]
        elif (modifier == 2): notes = [majorScales[rootIndex][0], majorScales[rootIndex][4]]
        else: notes = [majorScales[rootIndex][0], majorScales[rootIndex][2], majorScales[rootIndex][4]]
    
    #minor scale - root, (3rd), 5th, (7th)
    if chordType == 'min':
        if (modifier == 1): notes = [minorScales[rootIndex][0], minorScales[rootIndex][2], minorScales[rootIndex][4], minorScales[rootIndex][6]]
        elif (modifier == 2): notes = [minorScales[rootIndex][0], minorScales[rootIndex][4]]
        else: notes = [minorScales[rootIndex][0], minorScales[rootIndex][2], minorScales[rootIndex][4]]
    
    #major scale - root, 3rd flatted, 5th flatted
    if chordType == 'dim':
        flat3rd = majorScales[rootIndex][2] - 1
        if flat3rd < 0:
            flat3rd = 11
        
        flat5th = majorScales[rootIndex][4] - 1
        if flat5th < 0:
            flat5th = 11

        notes = [majorScales[rootIndex][0], flat3rd, flat5th]

    return notes

#--------------------------------------------------------------------#
# MELODY ------------------------------------------------------------#
#--------------------------------------------------------------------#

def generateMelody(data):
    return ""

#--------------------------------------------------------------------#
# HARMONY -----------------------------------------------------------#
#--------------------------------------------------------------------#

def identifyChords(data, neutral):
    
    roots = []
    durations = []

    previousNote = None
    accumulatedDuration = 0

    #filter passing notes
    roots = []
    durations = []

    previousNote = None
    accumulatedDuration = 0

    for note in data["BASS"]:
        pitch = data["BASS"][note]['pitch']
        duration = data["BASS"][note]['duration']

        if duration < 8:
            accumulatedDuration += duration
        else:
            if previousNote is not None:
                probSimplify = np.random.rand()
                # Special case: check if two consecutive notes have duration 8
                if previousNote['duration'] == 8 and duration == 8 and probSimplify < 0.5:
                    # Double the duration of the previous note
                    roots.append(previousNote['pitch'])
                    durations.append(previousNote['duration'] * 2 + accumulatedDuration)
                    accumulatedDuration = 0
                    previousNote = None  # Skip the current note
                else:
                    roots.append(previousNote['pitch'])
                    durations.append(previousNote['duration'] + accumulatedDuration)
                    accumulatedDuration = 0
                    previousNote = {'pitch': pitch, 'duration': duration}
            else:
                previousNote = {'pitch': pitch, 'duration': duration}

    # Append the last note if it exists
    if previousNote is not None:
        roots.append(previousNote['pitch'])
        durations.append(previousNote['duration'] + accumulatedDuration)

    # Ensure arrays are of the same length
    if len(roots) > len(durations):
        durations.append(accumulatedDuration)

    
    key = punctuateKey(data, ["BASS"])
    chordProgression = []

    print(roots)

    for r in roots:
        prob5th = np.random.rand()
        if prob5th > 0.50: r = majorScales[r][4]

        chordMatch = False
        #try major
        chord = [majorScales[r][0], majorScales[r][2], majorScales[r][4]]
        for note in chord:
            if note not in majorScales[key]:
                break
            elif note in majorScales[key] and note == chord[-1]:
                chordProgression.append(chord)
                chordMatch = True

        if not chordMatch:
            #try minor
            chord = [minorScales[r][0], minorScales[r][2], minorScales[r][4]]
            for note in chord:
                if note not in majorScales[key]:
                    break
                elif note in majorScales[key] and note == chord[-1]:
                    chordProgression.append(chord)
                    chordMatch= True
            
        if not chordMatch:
            chord = [majorScales[key][0], majorScales[key][2], majorScales[key][4]]
            chordProgression.append(chord)
            

                
        #chordProgression.append([majorScales[r][0], majorScales[r][2], majorScales[r][4]])
        #try minor
        #elif any(set([minorScales[r][0], minorScales[r][2], minorScales[r][4]]).issubset(set(majorScales[key])) for key in majorScales):
        #    chordProgression.append([minorScales[r][0], minorScales[r][2], minorScales[r][4]])

    '''startChord = np.random.randint(1, 7)
    model = load_model('models/harmonyAndBass.h5')
    progressionIndexes = generateChordProgression(model, [startChord], len(roots))

    for i in range(len(roots)):
        if neutral: chordProgression.append(getChordNotes(key, progressionIndexes[i], 2))
        else:
            prob7th = np.random.rand()
            if prob7th > 0.9: chordProgression.append(getChordNotes(key, progressionIndexes[i], 1))
            else: chordProgression.append(getChordNotes(key, progressionIndexes[i], 0))'''

    print(chordProgression, durations)

    return key, chordProgression, durations

def identifyRoots(data):
    
    start = None
    
    durations = []
    progression = []
    
    chord = []

    for note in data["HARMONY"]:
        note_info = data["HARMONY"][note]
        
        if start is None or start != note_info["start"]:
            if chord:
                progression.append(chord)
            chord = [note_info["pitch"]]
            start = note_info["start"]
            durations.append(note_info["duration"])
        else:
            chord.append(note_info["pitch"])

    # Append the last chord if it exists
    if chord:
        progression.append(chord)
    
    #print(progression, durations)
    roots = []

    for chord in progression:
        chord_set = set(chord)
    
        for root in range(12):
            major_triad = {majorScales[root][0], majorScales[root][2], majorScales[root][4]}
            major_seventh = {majorScales[root][0], majorScales[root][2], majorScales[root][4], majorScales[root][6]}
            
            minor_triad = {minorScales[root][0], minorScales[root][2], minorScales[root][4]}
            minor_seventh = {minorScales[root][0], minorScales[root][2], minorScales[root][4], minorScales[root][6]}
            
            if chord_set == major_triad:
                #print(f'{root} Major Triad')
                roots.append(root)
            elif chord_set == major_seventh:
                #print(f'{root} Major Seventh')
                roots.append(root)
            elif chord_set == minor_triad:
                #print(f'{root} Minor Triad')
                roots.append(root)
            elif chord_set == minor_seventh:
                #print(f'{root} Minor Seventh')
                roots.append(root)
    print(roots, durations)
    return roots, durations


def punctuateKey(data, tracks):
    keyPoints = [0,0,0,0,0,0,0,0,0,0,0,0]

    for t in tracks:
        for n in data[t]:
            note = data[t][n]
            for i in majorScales:
                if note["pitch"] == i:
                    keyPoints[i] += 2
                if note["pitch"] in majorScales[i]:
                    keyPoints[i] += 1
    
    #find the signature with more points
    key = 0
    maxPoints = 0
    for i in keyPoints:
        if i > maxPoints:
            maxPoints = i
            key = keyPoints.index(i)
    
    return key


def calculateChordNumProb():
    probNumNotes = np.random.rand()
    
    if probNumNotes < 0.5: numChords = 4
    elif probNumNotes >= 0.5 and probNumNotes < 0.7: numChords = 3
    elif probNumNotes >= 0.7 and probNumNotes < 0.85: numChords = 2
    else: numChords = 5

    return numChords


# Generate new progressions
def generateChordProgression(model, seed, length):

    # Parameters
    num_chords = 7  # There are 7 different chords in a major key (1-7)
    max_len = 5  # Maximum length of seed
    
    generated = seed
    for _ in range(length - len(seed)):
        padded_seed = pad_sequences([generated], maxlen=max_len-1, padding='post', value=0)
        #prediction = model.predict(padded_seed)

        prediction = model.predict(padded_seed)[0, len(generated)-1, :]
        prediction[0] = 0  # Set the probability of '0' to zero
        prediction /= prediction.sum()  # Normalize to ensure it sums to 1
        prediction[0] = 0  # Set the probability of '0' to zero
        next_chord = np.random.choice(range(num_chords+1), p=prediction)
        #next_chord = np.argmax(prediction[0, len(generated)-1, :])

        generated.append(next_chord)
   
    return generated

def generateDurations(model, seed, length):
    num_chords = 64
    max_len = 5

    generated = seed

    for _ in range(length - len(seed)):
        padded_seed = pad_sequences([generated], maxlen=max_len-1, padding='post', value=0)
        #prediction = model.predict(padded_seed)

        prediction = model.predict(padded_seed)[0, len(generated)-1, :]
        prediction[0] = 0  # Set the probability of '0' to zero
        prediction /= prediction.sum()  # Normalize to ensure it sums to 1
        prediction[0] = 0  # Set the probability of '0' to zero
        next_duration = np.random.choice(range(num_chords+1), p=prediction)
        #next_chord = np.argmax(prediction[0, len(generated)-1, :])

        generated.append(next_duration)

    return generated


def generateHarmony(data, neutral):

    octave = 4
    checkMelody = False
    checkBass = False

    #check if melody and bass are present
    for trackName in data:
        if trackName == "MELODY":
            checkMelody = True
        if trackName == "BASS":
            checkBass = True


    #track is empty (or just drums) or just melody (and drums)
    if not checkBass:
        
        if not checkMelody: key = np.random.randint(0, 12)
        else: key = punctuateKey(data, ["MELODY"])

        startChord = np.random.randint(1, 7)
        numChords = calculateChordNumProb()

        model = harmonyBassModel
        durationModel = durationModels[numChords]

        chordProgression = []
        progressionIndexes = generateChordProgression(model, [startChord], numChords)

        for i in progressionIndexes:
            if neutral: 
                chordProgression.append(getChordNotes(key, i, 2))
            else:
                prob7th = np.random.rand()
                if prob7th > 0.9: chordProgression.append(getChordNotes(key, i, 1))
                else: chordProgression.append(getChordNotes(key, i, 0)) 
        
        durations = generateDurations(durationModel, [], numChords)
    
    ##track has only bass or bass and melody (with or without drums)
    else:
        key, chordProgression, durations = identifyChords(data, neutral)

    #build harmony in BLOOM format
    notes = {}
    index = 0
    durationSum = 0

    for p in chordProgression:
        for i in p:
            note = {}
            note['pitch'] = i
            note['duration'] = int(durations[index])
            note['octave'] = octave
            note['start'] = int(durationSum)
            notes[len(notes)] = note
        
        durationSum += durations[index]
        index+=1
    
    return {0: keys[key]+" major", 1: notes}

#--------------------------------------------------------------------#
# BASS --------------------------------------------------------------#
#--------------------------------------------------------------------#

def addPassingNotes(bassLine, durations):

    factors = [2,4,8]
    for i in range(len(bassLine)-1):
        probPassing = np.random.rand()
        if probPassing > 0.5 and durations[i] >= 16:
            passingNote = majorScales[bassLine[i]][4]
            bassLine.insert(i+1, passingNote)
            passingDurations = round(durations[i]/factors[np.random.randint(0,3)])
            durations[i] = durations[i] - passingDurations
            durations.insert(i+1, passingDurations)

    return bassLine, durations
        
def generateBass(data):

    octave = 2
    checkMelody = False
    checkHarmony = False

    #check if melody and harmony are present
    for trackName in data:
        if trackName == "MELODY":
            checkMelody = True
        if trackName == "HARMONY":
            checkHarmony = True


    #track is empty (or just drums) or just melody (and drums)
    if not checkHarmony:

        if not checkMelody: key = np.random.randint(0, 12)
        else: key = punctuateKey(data, ["MELODY"])

        startChord = np.random.randint(1, 7)
        numChords = calculateChordNumProb()

        model = harmonyBassModel
        durationModel = durationModels[numChords]

        bassLine = []
        progressionIndexes = generateChordProgression(model, [startChord], numChords)
        for i in progressionIndexes: bassLine.append(getChordNotes(key, i, 0)[0])    
        
        durations = generateDurations(durationModel, [], numChords)
    
    else:
        roots, durations = identifyRoots(data)
        bassLine = []

        for r in roots:
            prob5th = np.random.rand()
            if prob5th > 0.75: bassLine.append(majorScales[r][4])
            else: bassLine.append(r)

    #add passing notes
    bassLine, durations = addPassingNotes(bassLine, durations)

    #build bass line in BLOOM format
    notes = {}
    index = 0
    durationSum = 0

    for p in bassLine:
        note = {}
        note['pitch'] = p
        note['duration'] = int(durations[index])
        note['octave'] = octave
        note['start'] = int(durationSum)
        
        notes[len(notes)] = note
        
        durationSum += int(durations[index])
        index+=1

    return {0: keys[0]+" major", 1: notes}
    #return {0: keys[key]+" major", 1: notes}

#--------------------------------------------------------------------#
# DRUMS -------------------------------------------------------------#
#--------------------------------------------------------------------#

drumSeeds = {
    1: np.array(([1],[0],[0],[0],[0],[0])),
    2: np.array(([1],[0],[0],[1],[0],[0])),
    3: np.array(([1],[0],[1],[0],[0],[0])),
    4: np.array(([1],[0],[0],[0],[1],[0])),
    5: np.array(([1],[0],[0],[0],[0],[1])),
    6: np.array(([1],[0],[0],[1],[0],[0])),
    7: np.array(([1],[0],[1],[0],[0],[0])),
    8: np.array(([1],[0],[0],[1],[1],[0])),
    9: np.array(([1],[0],[1],[0],[1],[0])),
    10: np.array(([1],[0],[0],[1],[0],[1])),
    11: np.array(([1],[0],[1],[0],[0],[1])),
    12: np.array(([1],[1],[0],[0],[0],[0])),
    13: np.array(([0],[0],[0],[0],[1],[0])),
    14: np.array(([0],[0],[0],[0],[0],[1])),
    15: np.array(([0],[0],[0],[1],[0],[0])),
    16: np.array(([0],[0],[1],[0],[0],[0])),
    17: np.array(([0],[0],[0],[1],[1],[0])),
    18: np.array(([0],[0],[1],[0],[1],[0])),
    19: np.array(([0],[0],[0],[1],[0],[1])),
    20: np.array(([0],[0],[1],[0],[0],[1])),
    21: np.array(([0],[1],[0],[0],[0],[0])),
}

def generateDrums(data, threadResults):
    # Parameters
    octave = 0
    num_parts = 6  # 7 different drum parts
    length = 32  # Length of each sequence

    randomStyle = np.random.randint(1, 5)
    if randomStyle == 1: model = drumModels[1]
    elif randomStyle == 2: model = drumModels[2]
    elif randomStyle == 3: model = drumModels[3]
    else: model = drumModels[4]

    seed = drumSeeds[np.random.randint(1, 4)]
    generated = seed

    for _ in range(length - seed.shape[1]):
        padded_seed = np.pad(generated, ((0, 0), (0, length-1 - seed.shape[1])), 'constant')
        padded_seed = padded_seed.transpose((1, 0))  # reshape to (sequence_length-1, num_parts)
        padded_seed = padded_seed.reshape((1, padded_seed.shape[0], padded_seed.shape[1]))
        prediction = model.predict(padded_seed)[0, len(generated[0])-1, :]
        prediction = np.random.binomial(1, prediction).astype(int)
        generated = np.hstack((generated, prediction.reshape((num_parts, 1))))
    
    newDrums = generated
    
    probBrake = np.random.rand()

    if probBrake > 0.6 and randomStyle != 1 and randomStyle != 3:
        length = 16

        model = drumModels['brake']

        seed = drumSeeds[np.random.randint(4, 22)]
        generated = seed
        
        for _ in range(length - seed.shape[1]):
            padded_seed = np.pad(generated, ((0, 0), (0, length-1 - seed.shape[1])), 'constant')
            padded_seed = padded_seed.transpose((1, 0))  # reshape to (sequence_length-1, num_parts)
            padded_seed = padded_seed.reshape((1, padded_seed.shape[0], padded_seed.shape[1]))
            prediction = model.predict(padded_seed)[0, len(generated[0])-1, :]
            prediction = np.random.binomial(1, prediction).astype(int)
            generated = np.hstack((generated, prediction.reshape((num_parts, 1))))
        
        brake = generated
    else:
        brake = -1
        #replace the last 16 steps of newDrum, all of the 6 parts
        #print("Brake")
        #print(newDrums)
        #print("---")
        #print(newDrums[:,16:32])
        #print("---")
        #print(generated)
        #print("Brake")
    

    '''for _ in range(length - seed.shape[1]):
        padded_seed = np.pad(generated, ((0, 0), (0, length-1 - seed.shape[1])), 'constant')
        padded_seed = padded_seed.transpose((1, 0))  # reshape to (sequence_length-1, num_parts)
        padded_seed = padded_seed.reshape((1, padded_seed.shape[0], padded_seed.shape[1]))
        prediction = model.predict(padded_seed)[0, len(generated[0])-1, :]
        prediction = (prediction > 0.5).astype(int)
        generated = np.hstack((generated, prediction.reshape((num_parts, 1))))'''
    
    newRhythm = newDrums

    notes = {}
    start = 0

    for times in range(2):

        if (times == 1 and type(brake) != type(-1)):
            newRhythm[:,16:32] = brake
            note = {}
            note['pitch'] = 6
            note['duration'] = 1
            note['octave'] = octave
            note['start'] = 0
            notes[len(notes)] = note
            
        for i in range(len(newRhythm)):
            for j in range(len(newRhythm[i])):
                if newRhythm[i][j] == 1:
                    note = {}
                    note['pitch'] = i
                    note['duration'] = 1
                    note['octave'] = octave
                    note['start'] = j+start
                    notes[len(notes)] = note
        start = 32

    probCrash = np.random.rand()
    
    if probCrash > 0.5:
        note = {}
        note['pitch'] = 6
        note['duration'] = 1
        note['octave'] = octave
        note['start'] = 0
        notes[len(notes)] = note

    #print (newRhythm)
    threadResults.put({0: keys[0]+" major", 1: notes})
