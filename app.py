# LEGO sorter project
# Main controller app
# (c) lego-sorter team, 2022-2023

import sys
import json
import logging
import gevent
from absl import flags

from flask import Flask, render_template, session, Response
from flask_socketio import SocketIO, emit, disconnect

from lib.globals import *
from lib.image_dataset import *
from lib.dummy_controller import DummyController
from lib.camera import Camera

FLAGS = flags.FLAGS
del FLAGS.zoom_factor
flags.DEFINE_float('zoom_factor', 2.5, short_name='zf', help='ROI zoom factor')
flags.DEFINE_boolean('debug', False, help='Start with debug info')

logging.basicConfig(format='%(levelname)s: %(message)s')
logger = logging.getLogger('lego-sorter')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, async_mode='gevent')

controller = DummyController()
camera = Camera(controller)

# @app.before_first_request
# def setup():
#     pass

# REST endpoints
@app.route('/list/ports')
def ports_list():
    return Response(json.dumps(controller.get_ports()),  mimetype='application/json')


@app.route('/list/cameras')
def camera_list():
    ci = Camera.get_camera_indexes()
    return ci


@app.route('/video_feed')
def video_feed():
    camera.stop_video_stream()
    return Response(camera.start_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Home page. Simple UI for testing"""
    return render_template('index.html')


# websocket
def message_listener():
    while session['connected'] == 1:
        message = controller.get_next_message()
        if message is not None and 'message_type' in message.keys():
            msg_type = message['message_type']

            match msg_type:
                case "S":
                    emit('status', message)
                case "C":
                    emit('confirmation', message)
                case "E":
                    emit('error', message)
                case _:
                    logger.error(f"Incorrect message type {msg_type}")

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


@socketio.on('select_camera', namespace='/sorter')
def select_camera(message):
    data: dict = json.loads(message)
    if "camera_id" in data.keys():
        auto_exposure = 0
        exposure = -10

        if "auto_exposure" in data.keys():
            auto_exposure = data["auto_exposure"]

        if "exposure" in data.keys():
            exposure = data["exposure"]

        camera.reset_camera(data["camera_id"], auto_exposure, exposure)
    else:
        logger.error("Incorrect message for select_camera request")


@socketio.on('run', namespace='/sorter')
def sorter_run(message):
    camera.capture_background()
    controller.run()


@socketio.on('wait', namespace='/sorter')
def sorter_wait(message):
    controller.wait()


@socketio.on('clean', namespace='/sorter')
def sorter_clean(message):
    controller.clean()


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
    socketio.wsgi_server.close()

    logger.info("User interface disconnected")


if __name__ == '__main__':
    FLAGS(sys.argv)
    logger.setLevel(logging.DEBUG if FLAGS.debug else logging.INFO)
    socketio.run(app)
