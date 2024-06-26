from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from src.pipelines.predict_pipeline import PredictPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        predict_obj = PredictPipeline()
        dict = predict_obj.get_output(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        domain = dict['domain']
        found = dict['found']
        recommended = dict['recommended']
        render = ''
        if (len(found)==0) and (len(recommended)==0):
            render = 'n'
        elif len(found)==0:
            render = 'r'
        elif len(recommended) == 0:
            render = 'f'
        else:
            render = 'b'
        return render_template('index.html', form=form,domain=domain,found=found,recommended=recommended,render=render)
    return render_template('index.html', form=form, visibility='hidden')

if __name__ == '__main__':
    app.run(debug=True)