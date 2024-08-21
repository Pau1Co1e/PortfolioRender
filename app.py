import os
import datetime
import numpy as np
from PIL import Image
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFError
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.stats import linregress
from skimage.color import rgb2gray
from transformers import pipeline
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_wtf import csrf

# Flask app configuration
app = Flask(__name__)

# Constants
ALLOWED_REDIRECTS = {
    'index': 'index',
    'about_me': 'about_me',
    'experience': 'experience',
    'contact_me': 'contact_me',
    'download': 'download',
    'csrf_error': 'csrf_error',
    'fractal_result': 'fractal_result',
    'fractal': 'fractal',
    'chatbot': 'chatbot',
    'upload': 'upload',
    'financial': 'financial',
}

# App settings
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads/')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = True  # Set to True if using HTTPS in production
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

CORS(app)
csrf = csrf.CSRFProtect(app)
db = SQLAlchemy(app)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load the FAQ pipeline model
faq_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


@app.before_request
def create_tables():
    db.create_all()


@app.errorhandler(400)
def bad_request(error):
    return "Bad Request!", 400, error


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('csrf_error.html', reason=e.description), 400


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chatbot')
def chatbot():
    session['visit_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Chatbot page accessed at {session['visit_time']}")
    return render_template('chatbot.html')


@app.route('/chatbot-answer', methods=['POST'])
@csrf.exempt
def chatbot_answer():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400

        context = (
            "My name is Paul Coleman. I am an AI and ML Engineer focused on "
            "building innovative solutions in Artificial Intelligence and Machine Learning. "
            "Feel free to ask about my projects, experience, or anything AI/ML related."
        )
        result = faq_pipeline(question=question, context=context)
        answer = result.get('answer', 'Sorry, I could not find an answer.')
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": "An error occurred while processing your question"}), 500


@app.route('/about_me')
def about_me():
    return render_template('about_me.html')


@app.route('/contact_me')
def contact_me():
    return render_template('contact_me.html')


@app.route('/download')
def download():
    return render_template('download.html')


@app.route('/experience', methods=['GET', 'POST'])
def experience():
    if request.method == 'POST':
        # Handle the POST request here (e.g., form submission)
        pass
    return render_template('experience.html')


@app.route('/download_report')
def download_report():
    try:
        fractal_dimension = 1.5
        image_paths = {
            'original': os.path.join(app.config['UPLOAD_FOLDER'], 'original.png'),
            'grayscale': os.path.join(app.config['UPLOAD_FOLDER'], 'grayscale.png'),
            'binary': os.path.join(app.config['UPLOAD_FOLDER'], 'binary.png'),
            'analysis': os.path.join(app.config['UPLOAD_FOLDER'], 'fractal_dimension_analysis.png')
        }
        pdf_path = generate_report(fractal_dimension, image_paths)
        return send_file(pdf_path, as_attachment=True)

    except ValueError as e:
        logger.error(f'Error generating report: {e}')
        flash('An error occurred while generating the report.', 'danger')
        return safe_redirect('fractal')


@app.route('/fractal', methods=['GET', 'POST'])
def fractal():
    if request.method == 'POST':
        try:
            file = validate_and_save_file(request)
            fractal_dimension, image_paths = calculate_fractal_dimension(file)
            return render_template('fractal_result.html', fractal_dimension=fractal_dimension, image_paths=image_paths)
        except ValueError as e:
            logger.error(f'Error processing image: {e}')
            flash(str(e), 'danger')
            return safe_redirect('fractal')

    return render_template('fractal.html')


def validate_and_save_file(requests):
    """Validate the uploaded file and save it to the configured upload folder."""
    if 'file' not in requests.files:
        raise ValueError('No file part')

    file = requests.files['file']
    if file.filename == '':
        raise ValueError('No selected file')

    if not (file and allowed_file(file.filename)):
        raise ValueError('Invalid file format')

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def calculate_fractal_dimension(image_path):
    """Calculate the fractal dimension of an image and save relevant images."""
    try:
        image = load_image(image_path)
        image_gray, image_binary = process_image(image)

        # Perform box counting
        fractal_dimension, log_box_sizes, log_box_counts, intercept = perform_box_counting(image_binary)

        # Save images and analysis graph
        image_paths = save_images(image, image_gray, image_binary, fractal_dimension, log_box_sizes, log_box_counts,
                                  intercept)

        return fractal_dimension, image_paths

    except Exception as e:
        logger.error(f"Error calculating fractal dimension: {str(e)}")
        raise


def load_image(image_path):
    """Load an image from the given path."""
    logger.info(f"Loading image from {image_path}")
    return Image.open(image_path)


def process_image(image):
    """Convert image to grayscale and binary formats."""
    logger.info("Converting image to grayscale")
    image_gray = rgb2gray(np.array(image))
    logger.info("Converting image to binary")
    image_binary = image_gray < 0.5
    return image_gray, image_binary


def perform_box_counting(image_binary):
    """Perform box counting to estimate the fractal dimension."""
    min_box_size, max_box_size, n_sizes = 2, min(image_binary.shape) // 4, 10
    box_sizes, box_counts = box_count(image_binary, min_box_size, max_box_size, n_sizes)
    log_box_sizes, log_box_counts = np.log(box_sizes), np.log(box_counts)
    slope, intercept, _, _, _ = linregress(log_box_sizes, log_box_counts)
    fractal_dimension = -slope
    logger.info(f"Fractal dimension calculated: {fractal_dimension}")
    return fractal_dimension, log_box_sizes, log_box_counts, intercept


def save_images(image, image_gray, image_binary, fractal_dimension, log_box_sizes, log_box_counts, intercept):
    """Save the original, grayscale, binary images, and the fractal dimension analysis graph."""
    image_paths = {
        'original': 'original.png',
        'grayscale': 'grayscale.png',
        'binary': 'binary.png',
        'analysis': 'fractal_dimension_analysis.png'
    }
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_paths['original']))
    plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], image_paths['grayscale']), image_gray, cmap='gray')
    plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], image_paths['binary']), image_binary, cmap='binary')

    # Save the fractal analysis graph using the log values and intercept
    save_fractal_analysis_graph(log_box_sizes, log_box_counts, fractal_dimension, intercept, image_paths['analysis'])

    return image_paths


def save_fractal_analysis_graph(log_box_sizes, log_box_counts, fractal_dimension, intercept, analysis_path):
    """Save the graph showing the fractal dimension analysis."""
    plt.figure(figsize=(6, 6))
    plt.scatter(log_box_sizes, log_box_counts, c='blue', label='Data Points')
    plt.plot(log_box_sizes, fractal_dimension * log_box_sizes + intercept, 'r',
             label=f'Linear Fit (Dimension = {fractal_dimension:.2f})')
    plt.xlabel('Log(Box Size)')
    plt.ylabel('Log(Box Count)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], analysis_path))
    logger.info("Fractal dimension analysis saved.")


def generate_report(fractal_dimension, image_paths):
    """Generate a PDF report for the fractal dimension analysis."""
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fractal_report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 40, "Fractal Dimension Analysis Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Estimated Fractal Dimension: {fractal_dimension:.2f}")
    c.drawString(100, height - 120, "Images:")

    for i, (title, path) in enumerate(image_paths.items()):
        c.drawImage(path, 100 + (i % 2) * 220, height - 220 - (i // 2) * 200, width=200, preserveAspectRatio=True,
                    mask='auto')

    c.drawString(100, 50, "Generated by Fractal Dimension Calculator")
    c.save()
    return pdf_path


def box_count(img, min_box_size, max_box_size, n_sizes):
    """Box counting algorithm to compute the number of boxes containing parts of the image."""
    sizes = np.floor(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_sizes)).astype(int)
    unique_sizes = np.unique(sizes)
    counts = []

    for size in unique_sizes:
        covered_boxes = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
                                        np.arange(0, img.shape[1], size), axis=1)
        counts.append(np.count_nonzero(covered_boxes > 0))

    return unique_sizes, counts


def safe_redirect(endpoint):
    """Safe redirect to a predefined list of endpoints to avoid open redirect vulnerabilities."""
    if endpoint in ALLOWED_REDIRECTS:
        return redirect(url_for(ALLOWED_REDIRECTS[endpoint]))
    else:
        return redirect(url_for('index'))  # Default to the homepage for safety


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render provides the PORT variable; default to 5000 if not set
    app.run(host='0.0.0.0', port=port)
