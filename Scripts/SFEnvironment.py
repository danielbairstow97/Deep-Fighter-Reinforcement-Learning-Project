import socket
import sys
from PIL import Image
from PIL import ImageFile
from random import randint
import io
import numpy as np
import time
from tensorforce.environments.environment import Environment

HOST = 'gs709a-2774'  # The server's hostname or IP address
PORT = 65432        # The port used by the server
##movement = ['NA', 'Up', 'Down', 'Left', 'Right', 'Up,Right',
##           'Up,Left', 'Down,Right', 'Down,Left']
actions = ['NA', 'Down', 'Left', 'Right', 'Down,Right', 'Down,Left', 'A', 'B', 'L', 'R', 'X', 'Y', 'Down,A', 'Left,X', 'Right,X']
actionsFlip = ['NA', 'Down', 'Right', 'Left', 'Down,Left', 'Down,Right', 'A', 'B', 'L', 'R', 'X', 'Y', 'Down,A', 'Right,X', 'Left,X']


class SFIIEnvironment(Environment):
    
    def __init__(self, resultPath, logger=None, defaultSave='Zangief3Star'):
        """
        Arguments:
        resultPath: Path to save results of training to
        """
	self.start = 0
	self.end = 0
        self.fp = open(resultPath, 'a+')
        self.sock = socket.socket()
        self.sock.bind((HOST, PORT))
        self.terminal = 0
        self.HPDiff = 0
	self.P1X = 205
        self.winHistory = list()
        self._actions = {'Actions': {'type': 'int', 'num_actions': len(actions)}}
        self._states =  dict(shape=[160, 230, 3], type='float')
	self.stateMessage = ''
	self.pvp = False
	self.outbox = ''
	self.logging = logger
        self.flip = 0
	self.start = 0
	self.default = defaultSave + ','
	self.spam = 0
	self.logits = None
	np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

    def close(self):                
        self.fp.close()
	self.outbox = 'TERMINATE,'
        self.send()
        self.conn.close()


    def states(self):
        return self._states
        

    def reset(self, match=None):
	self.spam = 0
        self.HPDiff = 0
	self.P1X = 205
        #Send RESET message and wait for acknowledgement before reading state
	if match is not None:
	    self.default=match + ','
	    print("New Challenger: " + self.default)

	if self.flip:
	    self.flip = 0
	else:
	    self.flip = 1
	if self.flip:
	    print("Flipped mode")
	else:
	    print("Normal mode")

	self.outbox = 'RESET,' + self.default
	self.send()
	self.getState()
        return self.state

    def execute(self, action, logits=None):
        #Read in appropriate message for movement and attack actions
        if self.flip:
            message = actionsFlip[action['Actions']]
        else:
            message = actions[action['Actions']]
        self.outbox = ','.join(("ACTIONS,P1",message, ""))
        #Communicate with emulator to carry out action and retrieve game state
        self.send()

	while not self.getState():
	    print("Message error, trying again")
	    self.logging.info("Message error, resending.")
	    self.outbox = 'RESEND,'
	    self.send()

        #Calculate reward
        self.calculateReward(action['Actions'], message, logits)

        return self.state, self.terminal, self.reward

    def getState(self):
	try:
	    inbox = self.conn.recv(32768)
	except socket.timeout as error:
	    return False

        split = inbox.split(b'|')
	self.stateMessage = split[0]
	
	if len(split[1]) == 0:
	    appended = b''
	else:
	    appended = split[1]

        while appended[-7:-4] != b'END':
	    try:
                screenMessage = self.conn.recv(32768)
	    except socket.timeout as error:
		return False

            appended = appended + screenMessage

        array = bytearray(appended)
        byteImage = io.BytesIO(array)
    	img = Image.open(byteImage)

        cropped = img.crop((0, 30, 256, 216))
        try:
	    res = cropped.resize((230,160))
	except:
	    return False
        self.state = np.array(res)
        if self.flip:
            flippedState = self.state[:,::-1,:]
            #flippedState[0:25,25:205,:] = self.state[0:25,25:205,:]
	    #if not self.start:
		#res.save("OG.png")
		#img = Image.fromarray(flippedState, "RGB")
		#img.save("Flipped.png")
            self.state = flippedState
	self.state = self.state / 255.0
	return True

    def actions(self):
        return self._actions
    
    def seed(self, seed):


        return None

    def calculateReward(self, action, message, logits):

        """
        Calculates reward and whether terminal state has been reached
        using message format:
        P1HP,P2HP,P1Win,P2Win
        """
	
        #Split message and convert to ints
        split = self.stateMessage.split(',')
        
        self.P1HP = int(split[0])
        self.P2HP = int(split[1])
        self.P1Win = int(split[2])
        self.P2Win = int(split[3])
	P1XNew = int(split[4])
	P2X = int(split[5])
	
	dir = 0
	if P2X > P1XNew:
	    dir = 1
	else:
	    dir = -1

	#Fix byte overflow
	if self.P1Win:
	    self.P2HP = 0
	elif self.P2Win:
	    self.P1HP = 0

        #Find if a player has won
        if self.P1Win or self.P2Win:
            self.terminal = 1
            self.fp.write("%d," %self.P1Win)
        else:
            self.terminal = 0
	#self.reward = 0
        #Calculate change in Health gap to decide on reward
        HPDiff = self.P1HP - self.P2HP
	self.reward = (HPDiff - self.HPDiff)/10.0

	if abs(P1XNew - P2X) > 80 and P1XNew != self.P1X:
	    if message == "Right" and P1XNew > self.P1X:
		if dir > 0:
	            self.reward += dir / 5.0
		else:
		    self.reward += dir / 4.0
	    elif message == "Left" and P1XNew < self.P1X:
		if dir * -1 > 0:
	            self.reward += dir * -1 / 5.0
		else:
		    self.reward += dir * -1 / 4.0

	if self.reward == 0 and action > 5:
	    self.reward += -0.08

	#if self.reward > 0:
	#    self.reward = 1
	#elif self.reward < 0:
	#    self.reward = -1
	
	self.reward = self.reward * 20
	sys.stdout.write("\rLogits: ")
	mn = max(logits['Actions'][0,:])
	for logit in logits['Actions'][0,:]:
	    sys.stdout.write('{0:.3f}'.format(logit) + '  ')
    	sys.stdout.flush()
	self.logits = logits

	self.P1X = P1XNew

	if self.P1Win:
	    self.reward += 5.0
	elif self.P2Win:
	    self.reward -= 5.0
        self.HPDiff = HPDiff


    def send(self):
        #Encodes and sends message 
        self.conn.send(self.outbox.encode())


    def connect(self):
        #Waits for client emulator to connect
        print('Waiting for connection')
        self.sock.listen(0)
        self.conn, self.addr = self.sock.accept()
	self.conn.settimeout(60)
        print('Connected')
   

    def __str__(self):
        return 'Street Fighter II Turbo'
