import eventlet
import json
import logging

from flask import Flask, render_template, session, Response
from flask_socketio import SocketIO, emit, disconnect
from lib.controller import Controller
from lib.camera import Camera
from lib.pipe_utils import *


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, async_mode='eventlet',
                    logger=True, engineio_logger=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

controller = Controller()

ui_connected = False


@app.before_first_request
def setup():
    logger.info("Before first request called")


# REST endpoints
@app.route('/list/ports')
def ports_list():
    portsFound = Controller.get_ports()
    ports = {}

    numConnection = len(portsFound)

    for i in range(0, numConnection):
        port = portsFound[i]
        strPort = str(port)
        splitPort = strPort.split(' ')
        ports[splitPort[0]] = strPort[strPort.find('USB'):]

    return Response(json.dumps(ports),  mimetype='application/json')


@app.route('/list/cameras')
def camera_list():
    ci = Camera.get_camera_indexes()
    return ci


@app.route('/video_feed')
def video_feed():
    return Response(controller.camera.get_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Home page. Simple UI for testing"""
    return render_template('index.html')


# websocket
def status_sender():
    while True:
        status = {'connected': session['connected']}
        logger.info(status)
        emit('status', status)
        eventlet.sleep(1)


def status_callback(data):
    emit('status', {'data': data, 'connected': session['connected']})


def confirmation_callback(message):
    emit('confirmation', {'message': message})


def error_fallback(message):
    emit('error', {'message': message})


@socketio.on('connect', namespace="/sorter")
def sorter_connect(auth):
    logger.info("User interface connected")


@socketio.on('connect_hw', namespace='/sorter')
def sorter_start(message):
    session['connected'] = 1
    controller.connect(status_callback, confirmation_callback, error_fallback)

    logger.info("Sorter started")


@socketio.on('activate_camera', namespace='/sorter')
def sorter_start(message):
    controller.activate_camera()


@socketio.on('run', namespace='/sorter')
def sorter_run(message):
    controller.run()


@socketio.on('wait', namespace='/sorter')
def sorter_wait(message):
    controller.wait()


@socketio.on('disconnect_hw', namespace='/sorter')
def sorter_stop(message):
    session['connected'] = 0
    controller.stop()

    logger.info("Sorter stopped")


@socketio.on('disconnect', namespace="/sorter")
def sorter_disconnect():
    logger.info("User interface disconnected")


if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)
