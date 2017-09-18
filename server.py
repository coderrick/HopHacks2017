from flask import Flask
from flask import request
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import wave
import urllib2
import alpha
import cgi, cgitb 


nchannels = 2
sampwidth = 2
framerate = 8000
nframes = 100

async_mode = None
app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
counter = 0
soundOrder = ["truth1",
		"lie1",
		"lie2",
		"truth2",
		"truth3",
		"lie3",
		"truth4",
		"lie4",
		"truth5",
		"lie5"]
@app.route('/',methods = ['GET','POST'])
def main():
	global counter
	if request.method == 'GET':
		counter = 0
		return render_template('index.html', async_mode=socketio.async_mode)

@app.route('/instructions1',methods = ['GET','POST'])
def instructions1():
	global counter
	if request.method == 'GET':
		return render_template('instruction1.html', async_mode=socketio.async_mode)

@app.route('/instructions2',methods = ['GET','POST'])
def instructions2():
	global counter
	if request.method == 'GET':
		counter = 0
		return render_template('instruction2.html', async_mode=socketio.async_mode)

@app.route('/baseline',methods = ['GET','POST'])
def baseline():
	global counter
	if request.method == 'GET':
		counter = 0
		return render_template('BaseLineTest.html', async_mode=socketio.async_mode)
	if request.method == 'POST':
		blobData = request.data
		fileName = soundOrder[counter]
		counter+=1
		decoded = base64.decodestring(blobData)
		fh = open("dag.wav", "wb")
		fh.write(decoded)
		fh.close()
# 		form = cgi.FieldStorage()

		# with contextlib.closing(wave.open("dag.wav",'r')) as f:
# 			fn.write(f)
# 		fn.close()
# 			
		return "gotcha"
		#audio.writeframes(request.data)
@app.route('/finaltest',methods = ['GET','POST'])
def finaltest():
	if request.method == 'GET':
		return render_template('Test.html', async_mode=socketio.async_mode)

@socketio.on('runCollection')
def runThread(message):	
	print message	
	alpha.get_additional_input_from_user("dag.wav")

if __name__ == "__main__":
	socketio.run(app, debug=True)