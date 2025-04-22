import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from decouple import config
from model.intoxication_model import load_model, preprocess_image, preprocess_audio

app = Flask(__name__)
app.config['SECRET_KEY'] = config('SECRET_KEY', default='your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///submissions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

db = SQLAlchemy(app)

# Ensure upload folders exist
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'audio'), exist_ok=True)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    audio_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50))
    probability = db.Column(db.JSON)
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<Submission {self.id} - {self.prediction or "Error"}>'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_submission():
    try:
        # Validate file uploads
        if 'image' not in request.files or 'audio' not in request.files:
            flash('Both image and audio files are required.', 'error')
            return redirect(url_for('index'))

        image_file = request.files['image']
        audio_file = request.files['audio']

        if image_file.filename == '' or audio_file.filename == '':
            flash('Please select both an image and an audio file.', 'error')
            return redirect(url_for('index'))

        # Validate file types
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            flash('Image must be PNG or JPEG.', 'error')
            return redirect(url_for('index'))

        if not audio_file.filename.lower().endswith(('.wav', '.mp3')):
            flash('Audio must be WAV or MP3.', 'error')
            return redirect(url_for('index'))

        # Save files
        image_filename = secure_filename(image_file.filename)
        audio_filename = secure_filename(audio_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', image_filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio', audio_filename)

        image_file.save(image_path)
        audio_file.save(audio_path)

        # Preprocess inputs
        image = preprocess_image(image_path)
        spectrogram = preprocess_audio(audio_path)

        if image is None or spectrogram is None:
            flash('Error processing uploaded files.', 'error')
            submission = Submission(
                image_path=image_path,
                audio_path=audio_path,
                error_message='Failed to preprocess image or audio.'
            )
            db.session.add(submission)
            db.session.commit()
            return render_template('results.html', error=True)

        # Load model and predict
        model = load_model()
        prediction = model.predict([image, spectrogram], verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_names = ['Sober', 'Mildly Intoxicated', 'Highly Intoxicated']
        result = class_names[predicted_class]
        probabilities = prediction[0].tolist()

        # Save to database
        submission = Submission(
            image_path=image_path,
            audio_path=audio_path,
            prediction=result,
            probability=probabilities
        )
        db.session.add(submission)
        db.session.commit()

        # Render results
        return render_template('results.html',
                              result=result,
                              probabilities=probabilities,
                              submission_id=submission.id,
                              image_url=url_for('static', filename=f'uploads/images/{image_filename}'),
                              audio_url=url_for('static', filename=f'uploads/audio/{audio_filename}'))

    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        submission = Submission(
            image_path=image_path if 'image_path' in locals() else '',
            audio_path=audio_path if 'audio_path' in locals() else '',
            error_message=str(e)
        )
        db.session.add(submission)
        db.session.commit()
        return render_template('results.html', error=True)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=config('DEBUG', default=True, cast=bool))
