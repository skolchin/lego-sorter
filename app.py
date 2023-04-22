import gevent
from gevent.event import Event
import json
import logging

from flask import Flask, render_template, session, Response
from flask_socketio import SocketIO, emit, disconnect
from lib.controller import Controller
from lib.camera import Camera
from lib.pipe_utils import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, async_mode='gevent',
                    logger=True, engineio_logger=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

controller = Controller()
camera = Camera()

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
    return Response(camera.get_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Home page. Simple UI for testing"""
    return render_template('index.html')


# websocket
def message_listener():
    while session['connected'] == 1:
        message = controller.get_next_message()
        if message is not None and 'mtype' in message.keys():
            mtype = message['mtype']

            match mtype:
                case "S":
                    emit('status', message)
                case "C":
                    emit('confirmation', {'message': message})
                case "E":
                    emit('error', {'message': message})
                case _:
                    logger.error(f"Incorrect message type {mtype}")

        gevent.sleep(1)


@socketio.on('connect', namespace="/sorter")
def sorter_connect(auth):
    logger.info("User interface connected")


@socketio.on('connect_hw', namespace='/sorter')
def sorter_start(message):
    session['connected'] = 1
    controller.connect()

    logger.info("Sorter connected")
    message_listener()


@socketio.on('run', namespace='/sorter')
def sorter_run(message):
    camera.capture_background()
    controller.run()


@socketio.on('wait', namespace='/sorter')
def sorter_wait(message):
    controller.wait()


@socketio.on('disconnect_hw', namespace='/sorter')
def sorter_stop(message):
    session['connected'] = 0
    camera.stop()
    controller.disconnect()

    logger.info("Sorter stopped")


@socketio.on('disconnect', namespace="/sorter")
def sorter_disconnect():
    camera.stop()
    controller.disconnect()

    socketio.wsgi_server.stop()

    logger.info("User interface disconnected")


if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)
