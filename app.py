from flask import Flask, render_template, request, redirect, url_for
from YOLOv8.detection import detect_parking_slots

app = Flask(__name__)
slots_status = [False,False,False,False,False,False]  # Start with all full

@app.route('/')
def index():
    available_slots = slots_status.count(True)
    return render_template("index.html", available_slots=available_slots, slots=slots_status)

@app.route('/detect', methods=['POST'])
def detect():
    global slots_status
    slots_status = detect_parking_slots()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
