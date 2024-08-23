import os
import datetime
import logging
import numpy as np
import psutil
from PIL import Image
from PIL.Image import Resampling
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, CSRFError
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.stats import linregress
from skimage.color import rgb2gray
from transformers import pipeline
from werkzeug.utils import secure_filename

# Flask app configuration
app = Flask(__name__)

# Constants
ALLOWED_REDIRECTS = {
    'index': 'index',
    'about_me': 'about_me',
    'experience': 'experience',
    'contact': 'contact',
    'download': 'download',
    'csrf_error': 'csrf_error',
    'fractal_result': 'fractal_result',
    'fractal': 'fractal',
    'chatbot': 'chatbot',
    'upload': 'upload',
    'financial': 'financial',
    'download_generated_report': 'download_generated_report'
}

# App settings
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config.update(
    UPLOAD_FOLDER=os.path.join(app.root_path, 'static/uploads/'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # Limit file size to 16MB
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif', 'svg'},
    SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', 'sqlite:///site.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_COOKIE_SECURE=True,  # Set to True if using HTTPS in production
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Initialize extensions
csrf = CSRFProtect(app)

# Initialize database
db = SQLAlchemy(app)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load the FAQ pipeline model
faq_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device="cpu")


@app.before_request
def create_tables():
    db.create_all()


@app.after_request
def add_cors_headers(response):
    """Manually add CORS headers to the response."""
    response.headers['Access-Control-Allow-Origin'] = 'https://portfoliorender-p89i.onrender.com'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


@app.errorhandler(400)
def bad_request(error):
    return "Bad Request!", 400, error


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('csrf_error.html', reason=e.description), 400


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about_me')
def about_me():
    return render_template('about_me.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Process the form data
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Example: Send the data to an email or save it to a database
        # Implement your logic here

        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('contact'))

    # If GET request, simply render the form
    return render_template('contact.html')


@app.route('/download')
def download():
    return render_template('download.html')


@app.route('/experience', methods=['GET', 'POST'])
def experience():
    return render_template('experience.html')


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
        logger.info(f"Received data: {data}")

        if not data:
            logger.warning(f"No JSON data received")
            return jsonify({"error": "No data received"}), 400

        question = data.get('question')

        if not question:
            logger.warning(f"Empty question received")
            return jsonify({"error": "No question provided"}), 400

        context = (
            "My name is Paul Coleman. I am an AI and ML Engineer focused on "
            "building innovative solutions in Artificial Intelligence and Machine Learning. "
            "Feel free to ask about my projects, experience, or anything AI/ML related."
            "JOBS = 'Software Engineer Intern, January 2021 – December 2021 "
            "Computer Science Assistant Teacher, January 2020 – December 2020"
            "Utah Valley University College of Engineering and Technology, Orem, UT'"
            
            "EXPERIENCE = Led a small team in the development and enhancement of humanoid robotics, focusing on "
            "performance and reliability."
            "Implemented network security measures for the robotics networking systems, including firewalls and "
            "encryption, protecting sensitive information."
            "Developed autonomous system functionalities using Artificial Intelligence and Machine Learning for "
            "object and facial recognition recognition and natural language processing."
            "Attended and gave live demos at university STEM recruitment fairs with the robots.'"
            
            "'Effective team leader with strong communication skills and work ethic. Continuous self-improvement "
            "through self-learning and real world applications.' 'Proven record of meeting deadlines and achieving "
            "goals, while exceeding expectations.' "
            "Meticulously organized and focused.'"
            
            "'Dedicated and insightful scholar with a relentless pursuit of knowledge in Computer Science and "
            "Mathematics, fortified through a rigorous academic journey at Utah Valley University.'"
            "' Mastered a broad spectrum of computer science disciplines, from foundational principles to advanced "
            "topics in Artificial Intelligence, Machine Learning, and Robotics.'"
            "'Studied mathematics and statistics ranging from discrete mathematics, numerical analysis, "
            "and probabilities and statistical analysis, to calculus and linear algebra'"
            "EDUCATION = 'Master of Science in Computer Science, August 2025, Utah Valley University, Orem, UT, "
            "GPA 3.3; Bachelor of Science in Computer Science, August 2022, Utah Valley University, Orem, UT, "
            "GPA 3.4; CERTIFICATIONS/ACHIEVEMENTS = Programmer, August 2020 Utah Valley University, Orem, "
            "UT; Deans List, Utah Valley University, Orem, UT'"
            
            "SKILLS = 'Database Management: Microsoft SQL Server, MySQL, Data Analysis, Database Design, RDBM.'"
            "SKILLS = 'Programming: Python, Unity C#, Java, PyTorch, TensorFlow, sklearn, JSON, XML.'"
            "SKILLS = 'Artificial Intelligence and Machine Learning: Model Development, Algorithm Design, "
            "Accuracy Improvements, Deep Fake Detection, Anti-Spam/Phishing Intelligent Email Filter, Autonomous "
            "Robotics Systems, Computer Vision, Facial Image Detection, Natural Language Processing, "
            "Sentiment Analysis.'"
        )
        result = faq_pipeline(question=question, context=context)
        answer = result.get('answer', 'Sorry, I could not find an answer.')
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": "An error occurred while processing your question"}), 500


@app.route('/fractal', methods=['GET', 'POST'])
@csrf.exempt
def fractal():
    if request.method == 'POST':
        try:
            file_path = validate_and_save_file(request)
            fractal_dimension, image_paths = calculate_fractal_dimension(file_path)

            # Generate PDF report
            pdf_path = generate_report(fractal_dimension, image_paths)
            # session['pdf_path'] = pdf_path
            logger.info(f"Fractal dimension calculated: {fractal_dimension}")

            # Render result template with the download link
            return render_template('fractal_result.html', fractal_dimension=fractal_dimension, image_paths=image_paths, pdf_path=pdf_path)

        except ValueError as e:
            logger.error(f'Error processing image: {e}')
            flash(str(e), 'danger')
            return safe_redirect('fractal')

    return render_template('fractal.html')


def validate_and_save_file(request):
    """Validate the uploaded file and save it to the configured upload folder."""
    if 'file' not in request.files:
        raise ValueError('No file part')

    file = request.files['file']
    if file.filename == '':
        raise ValueError('No selected file')

    if not allowed_file(file.filename):
        raise ValueError('Invalid file format')

    filename = secure_filename(file.filename)
    upload_folder = app.config['UPLOAD_FOLDER']

    # Ensure the upload folder exists
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    return file_path


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def calculate_fractal_dimension(image_path):
    """Calculate the fractal dimension of an image and save relevant images."""
    try:
        image = Image.open(image_path)
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


def resize_image(image_path, max_size=(1024, 1024)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.LANCZOS)
        resized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + os.path.basename(image_path))
        img.save(resized_path)
        return resized_path


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
    try:
        image_paths = {
            'original': 'uploads/original.png',
            'grayscale': 'uploads/grayscale.png',
            'binary': 'uploads/binary.png',
            'analysis': 'uploads/fractal_dimension_analysis.png'
        }
        image.resize((200, 200), resample=Resampling.LANCZOS)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'original.png'))
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], 'grayscale.png'), image_gray, cmap='gray')
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], 'binary.png'), image_binary, cmap='binary')

        # Save the fractal analysis graph
        save_fractal_analysis_graph(log_box_sizes, log_box_counts, fractal_dimension, intercept,
                                    os.path.join(app.config['UPLOAD_FOLDER'], ' fractal_dimension_analysis.png'))

        logger.info(f"Images saved successfully: {image_paths}")
        return image_paths

    except Exception as e:
        logger.error(f"Error saving images: {str(e)}")
        raise


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
    plt.close()
    logger.info("Fractal dimension analysis saved.")


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory Usage: {memory_info.rss / (1024 * 1024)} MB")


def generate_report(fractal_dimension, image_paths):
    """Generate a PDF report for the fractal dimension analysis."""
    try:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fractal_report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, height - 40, "Fractal Dimension Analysis Report")

        c.setFont("Helvetica", 12)
        c.drawString(100, height - 80, f"Estimated Fractal Dimension: {fractal_dimension:.2f}")
        c.drawString(100, height - 120, "Images:")
        c.drawString(x=100, y=100, text=" ")
        image_width, image_height = 200, 200  # Fixed size for images

        for i, (title, pdf_path) in enumerate(image_paths.items()):
            if os.path.exists(pdf_path):
                logger.info(f"Adding image {title} to PDF: {pdf_path}")
                c.drawImage(
                    pdf_path,
                    100 + (i % 2) * 220,
                    height - 220 - (i // 2) * 200,
                    width=image_width,
                    height=image_height,
                    preserveAspectRatio=True,
                    mask='auto'
                )
            else:
                logger.error(f"Image not found: {pdf_path}")

        c.drawString(100, 50, "Generated by Fractal Dimension Calculator")
        c.save()
        logger.info(f"PDF generated successfully: {pdf_path}")
        return pdf_path

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise


@app.route('/download_generated_report')
def download_generated_report():
    pdf_path = request.args.get('pdf_path')
    if pdf_path or os.path.exists(pdf_path):
        flash('Report not found.', 'danger')
        return safe_redirect('fractal')

    return send_file(pdf_path, as_attachment=True)


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
    app.run(debug=True, host='0.0.0.0', port=port)
