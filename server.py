
from threading import Lock

from flask import Flask, render_template
from flask_socketio import SocketIO

from AudioAnalysis.StringStreamer import StringStreamer


# CONSTANTS

PORT = 8080

# Init Flask App
server = Flask(__name__)

# Init Socket 
socket = SocketIO(server, async_mode=None, cors_allowed_origins="*")

# Threading Vars
thread = None
thread_lock = Lock()



# Index Route
@server.route('/')
def index(): 
    return 'Hello World!'


# Socket Connection
@socket.event
def connect():

    global thread
    with thread_lock:
        if thread is None:
            thread = socket.start_background_task(backgroundThread)

    print("Client Connected!")


# Socket Disconnect
@socket.on('disconnect')
def test_disconnect():
    print('Client Disconnected!')



def stringDataCallback(frequency, stringId, timeStamp):

    print("Frequency: ", frequency)
    print("Time Stamp: ", timeStamp)

    socket.emit('newTabData',
                      {'data': {'frequency': frequency, 'stringId': stringId, 'time': timeStamp}})


# Background thread for audio processing!
def backgroundThread():

    print("STARTING BACKGROUND THREAD")

    # Run the StringStreamer!
    stringStreamer = StringStreamer("Flask String Stream", stringDataCallback)

    count = 0
    while True:
        socket.sleep(1)
        count += 1
        # socket.emit('newTabData',
        #               {'data': 'Server generated event', 'count': count})




if __name__ == '__main__':

    socket.run(server, port=PORT, debug= True)
    
    




