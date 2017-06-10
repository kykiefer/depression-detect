from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/donate', methods=['GET', 'POST'])
def donate():
    return render_template('donate.html')

@app.route('/contact')
def results():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
