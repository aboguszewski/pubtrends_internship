import os
from flask import Flask, render_template, request
from dataset_clustering import generate_html_cluster_plot

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# TODO: CURRENTLY 3MIN 53S ON FULL LIST
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file and file.filename.endswith('.txt'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            script, div, cdn_jss = generate_html_cluster_plot(file_path)
            return render_template('plot.html', script=script, div=div, cdn_jss=cdn_jss)

        return "Invalid file type. Please upload a .txt file.", 400

    return render_template('upload.html')


if __name__ == '__main__':
    app.run()
