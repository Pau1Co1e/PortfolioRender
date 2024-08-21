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
from flask_wtf.csrf import CSRFProtect

# Flask app configuration
app = Flask(__name__)

CORS(app)

csrf = CSRFProtect(app)

# List of allowed redirects
ALLOWED_REDIRECTS = {
    'index': 'index',
    'about_me': 'about_me',
    'experience': 'experience',
    'contact_me': 'contact_me',
    'download': 'download',
    'csrf_error': 'csrf_error',
    'fractal_result': 'fractal_result',
    'fractal': 'fractal',
    'chatbot': 'chatbot'
}

app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
# app.secret_key = os.environ.get('SECRET_KEY')
# if app.secret_key == 'default_secret_key':
#     raise ValueError("The SECRET_KEY is not set properly in the environment!")

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads/')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# Use DATABASE_URL environment variable if available, fallback to SQLite for local and Render environments
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///site.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load the FAQ pipeline model
faq_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


@app.before_request
def create_tables():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chatbot')
def chatbot():
    session['visit_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Chatbot page accessed at {session['visit_time']}")
    return render_template('chatbot.html')


@app.route('/chatbot-answer', methods=['POST'])
@csrf.exempt  # Ensure CSRF protection is explicitly managed
def chatbot_answer():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        if not data:
            logger.warning("No JSON data received")
            return jsonify({"error": "No data received"}), 400

        question = data.get('question', '')

        if not question:
            logger.warning("Empty question received")
            return jsonify({"error": "No question provided"}), 400

        context = ("My name is Paul Coleman. I am an AI and ML Engineer focused on "
                   "building innovative solutions in Artificial Intelligence and Machine Learning. "
                   "Feel free to ask about my projects, experience, or anything AI/ML related.")

        result = faq_pipeline(question=question, context=context)
        answer = result.get('answer', 'Sorry, I could not find an answer.')

        logger.info(f"Question: {question} | Answer: {answer}")
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
    fractal_dimension = 1.5
    image_paths = {
        'original': os.path.join(app.config['UPLOAD_FOLDER'], 'original.png'),
        'grayscale': os.path.join(app.config['UPLOAD_FOLDER'], 'grayscale.png'),
        'binary': os.path.join(app.config['UPLOAD_FOLDER'], 'binary.png'),
        'analysis': os.path.join(app.config['UPLOAD_FOLDER'], 'fractal_dimension_analysis.png')
    }

    try:
        pdf_path = generate_report(fractal_dimension, image_paths)
    except ValueError as e:
        logger.error(f'Error generating report: {e}')
        flash('An error occurred while generating the report.', 'danger')
        return safe_redirect('fractal')

    return send_file(pdf_path, as_attachment=True)


@app.errorhandler(400)
def bad_request(error):
    return "Bad Request!", 400, error


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('csrf_error.html', reason=e.description), 400


def generate_report(fractal_dimension, image_paths):
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fractal_report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 40, "Fractal Dimension Analysis Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Estimated Fractal Dimension: {fractal_dimension:.2f}")

    c.drawString(100, height - 120, "Images:")
    c.drawImage(image_paths['original'], 100, height - 220, width=200, preserveAspectRatio=True, mask='auto')
    c.drawImage(image_paths['grayscale'], 320, height - 220, width=200, preserveAspectRatio=True, mask='auto')
    c.drawImage(image_paths['binary'], 100, height - 400, width=200, preserveAspectRatio=True, mask='auto')
    c.drawImage(image_paths['analysis'], 320, height - 400, width=200, preserveAspectRatio=True, mask='auto')

    c.drawString(100, 50, "Generated by Fractal Dimension Calculator")

    c.save()
    return pdf_path


# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/fractal', methods=['GET', 'POST'])
def fractal():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return safe_redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return safe_redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            fractal_dimension, image_paths = calculate_fractal_dimension(image_path)
            return render_template('fractal_result.html', fractal_dimension=fractal_dimension, image_paths=image_paths)

    return render_template('fractal.html')


def calculate_fractal_dimension(image_path):
    try:
        logger.info(f"Processing image at: {image_path}")

        # Open the image
        image = Image.open(image_path)
        logger.info("Image opened successfully.")

        # Convert to grayscale
        image_gray = rgb2gray(np.array(image))
        logger.info("Image converted to grayscale.")

        # Convert to binary image using a threshold
        threshold = 0.5
        image_binary = image_gray < threshold
        logger.info("Image converted to binary.")

        # Parameters for box counting
        min_box_size = 2
        max_box_size = min(image_binary.shape) // 4
        n_sizes = 10

        # Perform box counting
        box_sizes, box_counts = box_count(image_binary, min_box_size, max_box_size, n_sizes)
        logger.info(f"Box sizes: {box_sizes}")
        logger.info(f"Box counts: {box_counts}")

        # Calculate the fractal dimension
        log_box_sizes = np.log(box_sizes)
        log_box_counts = np.log(box_counts)
        slope, intercept, r_value, p_value, std_err = linregress(log_box_sizes, log_box_counts)
        fractal_dimension = -slope
        logger.info(f"Fractal dimension calculated: {fractal_dimension}")

        # Save the images
        image_paths = {
            'original': 'original.png',
            'grayscale': 'grayscale.png',
            'binary': 'binary.png',
            'analysis': 'fractal_dimension_analysis.png'
        }

        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_paths['original']))
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], image_paths['grayscale']), image_gray, cmap='gray')
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], image_paths['binary']), image_binary, cmap='binary')

        # Plot and save the analysis graph
        plt.figure(figsize=(6, 6))
        plt.scatter(log_box_sizes, log_box_counts, c='blue', label='Data Points')
        plt.plot(log_box_sizes, fractal_dimension * log_box_sizes + intercept, 'r',
                 label=f'Linear Fit (Dimension = {fractal_dimension:.2f})')
        plt.xlabel('Log(Box Size)')
        plt.ylabel('Log(Box Count)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], image_paths['analysis']))
        logger.info("Fractal dimension analysis saved.")

        return fractal_dimension, image_paths

    except Exception as e:
        logger.error(f"Error calculating fractal dimension: {str(e)}")
        raise  # Re-raise the exception for further handling if needed


def box_count(img, min_box_size, max_box_size, n_sizes):
    sizes = np.floor(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_sizes)).astype(int)
    unique_sizes = np.unique(sizes)
    counts = []

    for size in unique_sizes:
        covered_boxes = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
            np.arange(0, img.shape[1], size), axis=1
        )
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
