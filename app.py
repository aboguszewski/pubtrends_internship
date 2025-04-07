import os
from flask import Flask, render_template, request
from dataset_clustering import generate_html_cluster_plot


app = Flask(__name__, template_folder='templates', static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    plot_components = None
    error_message = None
    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = "No file part."
        else:
            file = request.files['file']
            if file.filename == '':
                error_message = "No selected file."
            elif file and file.filename.endswith('.txt'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                try:
                    script, div, cdn_jss = generate_html_cluster_plot(file_path)
                    plot_components = {'script': script, 'div': div, 'cdn_jss': cdn_jss}
                except Exception as e:
                    error_message = f"Error: {e}"
            else:
                error_message = "Invalid file type. Please upload a .txt file."
    return render_template('index.html', plot=plot_components, error=error_message)


if __name__ == '__main__':
    app.run()
