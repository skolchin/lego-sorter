from absl import app
from lib.controller import Controller


def main(_):
    controller = Controller()
    print(controller.find_controller())


if __name__ == '__main__':
    app.run(main)
