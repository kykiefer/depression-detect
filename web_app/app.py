from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/donate')
def donate():
    return render_template('donate.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
