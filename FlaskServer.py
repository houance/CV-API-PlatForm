from flask import Flask, jsonify
from Utils.configReader import configReader

app = Flask(__name__)
reader = configReader('Router/AddressTable.ini')
values = reader.readValues()


@app.route('/yolo', methods=['POST'])
def yoloAddress():
    return jsonify(values[0][0])


@app.route('/haar', methods=['POST'])
def haarAddress():
    return jsonify(values[1][0])


@app.route('/yuNet', methods=['POST'])
def yuNetAddress():
    return jsonify(values[2][0])


if __name__ == '__main__':
    app.run('127.0.0.1', 5000, debug=True)
