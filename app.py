import datetime
from flask import (
    Flask, render_template, request, jsonify, flash, redirect, url_for,
    send_file, session, send_from_directory, Response, g
)
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from flask_wtf.csrf import CSRFProtect
import logging
from logging.handlers import RotatingFileHandler
from matplotlib import pyplot as plt
import mimetypes
import numpy as np
import os
from pythonjsonlogger import jsonlogger  # JSON logging
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import secrets
from scipy.stats import linregress
from urllib.parse import unquote
import uuid
from werkzeug.utils import secure_filename
from flask_cors import CORS
import redis

# Initialize Flask app
app = Flask(__name__)

# Configure JSON logging
logger = logging.getLogger("flask_app")
logger.setLevel(logging.INFO)

# Remove all existing handlers to prevent duplication
if not logger.hasHandlers():
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s %(funcName)s %(lineno)d')
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

# Optionally suppress Flask's default logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Only log errors from Flask's internal logger

# Configuration
PORT = int(os.getenv("PORT", 8000))
DEBUG = False

# Secret Key
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    logger.error("No SECRET_KEY set for Flask application.")
    raise ValueError("No SECRET_KEY set for Flask application.")
app.secret_key = SECRET_KEY

# Environment-based Configuration
env = app.config.get("ENV", "development")
if env == "production":
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv('DATABASE_URL')
    if not app.config["SQLALCHEMY_DATABASE_URI"]:
        logger.error("DATABASE_URL not set for production environment.")
        raise ValueError("DATABASE_URL not set for production environment.")
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///site.db'

# Session Configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_SECURE'] = env == 'production'
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # Limit file size to 16MB
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif'},
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Define upload and video folders with consistent defaults
# Override environment variables if they point to read-only directories
upload_folder = os.getenv('UPLOAD_FOLDER', '/tmp/uploads')
video_folder = os.getenv('VIDEO_FOLDER', '/tmp/videos')

if upload_folder.startswith('/app'):
    logger.warning("UPLOAD_FOLDER is set to a read-only directory '/app'. Overriding to '/tmp/uploads'.")
    upload_folder = '/tmp/uploads'

if video_folder.startswith('/app'):
    logger.warning("VIDEO_FOLDER is set to a read-only directory '/app'. Overriding to '/tmp/videos'.")
    video_folder = '/tmp/videos'

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['VIDEO_FOLDER'] = video_folder

# Create necessary directories
def create_directories():
    upload_folder = app.config['UPLOAD_FOLDER']
    video_folder = app.config['VIDEO_FOLDER']
    folders = [upload_folder, video_folder]
    for folder_path in folders:
        try:
            os.makedirs(folder_path, exist_ok=True)
            os.chmod(folder_path, 0o755)  # Set appropriate permissions
            logger.info({
                "message": f"Directory created or already exists: {folder_path}",
                "folder_path": folder_path
            })
        except OSError as e:
            logger.error({
                "message": f"Failed to create directory {folder_path}",
                "error": str(e)
            })
            raise

create_directories()

# Initialize Extensions
csrf = CSRFProtect(app)
db = SQLAlchemy(app)

# Initialize Flask-Caching with Redis for Production and Simple Cache for Development
if env == "production":
    cache_config = {
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.getenv('REDIS_URL')  # Ensure this env variable is set
    }
else:
    cache_config = {
        'CACHE_TYPE': 'simple'
    }
cache = Cache(app, config=cache_config)

# Initialize Flask-Limiter with Redis storage
redis_url = os.getenv("REDIS_URL")
if env == "production" and not redis_url:
    logger.error("REDIS_URL environment variable not set for production.")
    raise ValueError("REDIS_URL environment variable not set for production.")

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=redis_url if env == "production" else None,  # Use Redis only in production
    default_limits=["200 per day", "50 per hour"]
)

# Initialize Redis client (for general use)
if env == "production":
    FASTAPI_URL = os.getenv('FASTAPI_URL', 'http://127.0.0.1:8000/faq/')

    if not FASTAPI_URL:
        logger.error("FASTAPI_URL environment variable not set for production.")
        raise ValueError("FASTAPI_URL environment variable not set for production.")

    try:
        redis_pool = redis.ConnectionPool.from_url(redis_url, socket_timeout=5)
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.ping()
        logger.info({"message": "Connected to Redis successfully."})
    except redis.exceptions.RedisError as e:
        logger.error({"message": "Failed to connect to Redis.", "error": str(e)})
        redis_client = None
if env == 'development':

    FASTAPI_URL = "https://api.codebloodedfamily.com/faq/"
else:
    FASTAPI_URL = os.getenv('FASTAPI_URL')  # Optional in development
    redis_client = None  # No Redis in development

# Initialize a global session with retries and connection pooling
global_session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.3,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"]
)
adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
global_session.mount("https://", adapter)
global_session.mount("http://", adapter)

# Configure CORS using flask_cors.CORS exclusively
CORS(app, resources={r"/*": {"origins": ["https://codebloodedfamily.com"]}}, supports_credentials=True)

# Constants
ALLOWED_REDIRECTS = {
    'index', 'about_me', 'experience', 'contact', 'download', 'csrf_error',
    'fractal', 'chatbot', 'upload', 'download_generated_report', 'uploaded_file'
}

# Flask routes
@app.before_request
def before_request():
    """Actions to perform before each request."""
    # Generate a unique nonce for each request and store it in the global `g` object.
    g.nonce = secrets.token_hex(16)  # Generates a 32-character hexadecimal string
    logger.debug(f"Generated nonce: {g.nonce}", extra={'action': 'nonce_generated'})


# Inject nonce into templates
@app.context_processor
def inject_nonce():
    """Inject the nonce into the template context."""
    return dict(nonce=getattr(g, 'nonce', ''))


@app.after_request
def after_request(response):
    """Set security headers after each request."""
    # Add Content Security Policy header with nonce
    nonce = getattr(g, 'nonce', '')
    response.headers['Content-Security-Policy'] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}' https://code.jquery.com https://cdn.jsdelivr.net; "
        f"style-src 'self' 'nonce-{nonce}' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data:; "
        "connect-src 'self' https://api.huggingface.co; "
        "frame-src 'self' https://docs.google.com; "
        "font-src 'self' https://cdn.jsdelivr.net; "
    )

    # Apply Security Headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    response.headers['Referrer-Policy'] = 'no-referrer-when-downgrade'
    response.headers['Permissions-Policy'] = "geolocation=(), microphone=(), camera=()"

    return log_response(response)


@app.errorhandler(400)
def handle_bad_request(error):
    if hasattr(error, 'description') and 'CSRF' in error.description:
        logger.error(f"CSRF Error: {error.description}", extra={'action': 'csrf_error'})
        return render_template('csrf_error.html', reason=error.description), 400
    logger.error(f"Bad Request: {error}", extra={'action': 'bad_request'})
    return "Bad Request", 400


@app.errorhandler(404)
def page_not_found(error):
    logger.error(f"Page not found: {error}", extra={'action': 'page_not_found'})
    return render_template('404.html'), 404


@app.errorhandler(429)
def ratelimit_handler(error):
    logger.error(f"Error 429: {error}", extra={'action': 'ratelimit_handler'})
    return jsonify(error="Rate limit exceeded. Please try again later."), 429


@app.errorhandler(500)
def handle_server_error(error):
    logger.error(f"Server Error: {error}", extra={'action': 'server_error'})
    return "Internal Server Error", 500


@app.route('/')
def index():
    if DEBUG is True:
        logger.info("Rendered home page", extra={'action': 'render_page', 'page': 'index'})
    return render_template('index.html')


@app.route('/test-faq', methods=['GET'])
def test_faq():
    try:
        response = global_session.post(
            FASTAPI_URL,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error during test_faq: {http_err}", extra={'action': 'test_faq_http_error'})
        return jsonify({"error": "HTTP error occurred during test_faq."}), 500
    except Exception as error:
        logger.error(f"Error during test_faq: {error}", extra={'action': 'test_faq_error'})
        return jsonify({"error": "An error occurred during test_faq."}), 500


@app.route('/about_me')
def about_me():
    if DEBUG:
        logger.info("Rendered about_me page", extra={'action': 'render_page', 'page': 'about_me'})
    return render_template('about_me.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash('Your message has been sent successfully!', 'success')
        if DEBUG:
            logger.info('Contact form submitted', extra={'action': 'contact_form_submitted'})
        return redirect(url_for('contact'))
    return render_template('contact.html')


@app.route('/download')
def download():
    if DEBUG:
        logger.info("Rendered download page", extra={'action': 'render_page', 'page': 'download'})
    return render_template('download.html')


@app.route('/experience', methods=['GET', 'POST'])
def experience():
    if DEBUG:
        logger.info("Rendered experience page", extra={'action': 'render_page', 'page': 'experience'})
    return render_template('experience.html')


@app.route('/chatbot')
def chatbot():
    session['visit_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if DEBUG:
        logger.info(f"Chatbot page accessed at {session['visit_time']}", extra={'action': 'chatbot_accessed'})
    return render_template('chatbot.html')


@app.route('/chatbot-answer', methods=['POST'])
@csrf.exempt  # Exempt this route from CSRF protection
@limiter.limit("10 per minute")
def chatbot_answer():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logger.warning({
                "message": "No question provided in chatbot request"
            })
            return jsonify({"error": "No question provided"}), 400

        # Preprocess the question
        try:
            question = preprocess_question(data['question'])
        except ValueError as ve:
            logger.warning(f"Invalid question: {ve}", extra={'action': 'chatbot_error'})
            return jsonify({"error": "Invalid question provided."}), 400

        # Check if the response is cached
        cached_response = cache.get(f"chatbot_answer_{question}")
        if cached_response:
            if DEBUG:
                logger.info("Returning cached response", extra={'action': 'cached_response_returned'})
            return jsonify(cached_response)

        # Call FastAPI service to handle the model inference
        response = call_faq_pipeline(question)

        # Cache the response if it's valid
        if response and 'answer' in response and len(response['answer']) > 3:
            cache.set(f"chatbot_answer_{question}", response, timeout=60 * 5)

        if DEBUG:
            logger.info("Chatbot successfully answered question", extra={'action': 'chatbot_answered'})

        return jsonify(response)

    except Exception as error:
        logger.error(f"Error processing chatbot request: {str(error)}", extra={'action': 'chatbot_error'})
        return jsonify({"error": "An error occurred while processing your question."}), 500


def call_faq_pipeline(question):
    # Static context can be abstracted to its own function or module if it grows
    static_context = (
        "My name is Paul Coleman. I'm currently enrolled as a graduate student in the computer science masters "
        "program at Utah Valley University."
        "I am working towards becoming an AI/ML Engineer with an interest in applying those skills to Finance, "
        "Cybersecurity, or Healthcare sectors. "
        "I have 5 years of programming experience with Python and AI/ML frameworks TensorFlow and PyTorch."
        "Most Recent Professional Work Experience or Job Title: Full Stack Web Developer."
        "Tools and programming languages that I used when I was working as a full stack web developer: C#, "
        "ASP.NET Core."
        "Earned a Bachelors degree in Computer Science on August 2022 from Utah Valley University."
        "I will be graduating with a masters degree in Computer Science from Utah Valley University in the Fall "
        "of 2025."
        "AI related skills and expertise that I have are mathematics and statistics, discrete mathematics, "
        "numerical analysis, probabilities and statistical analysis, data analysis, calculus, and linear algebra. "
        "Full Stack Developer Skills and Experience: Flask, C#, PHP, Search Engine Optimization, User Experience, "
        "Design, User Interface Design, Responsive Design, Postgres, Git, Swift. "
        "Programming Languages: Python, C#, Java, SQL, HTML5/CSS3, JavaScript."
        "My email is engineering.paul@icloud.com."
        "Completed an internship as a robotics software engineer. The internship was my introduction to machine "
        "learning and artificial intelligence and how to apply it in the real world. As an intern, I developed "
        "computer vision, natural language processing, and autonomous functionalities for humanoid robots."
    )

    # Combine static context with the question
    context = f"{static_context} {question.strip()}"

    # Payload structure to send to FastAPI, ensuring the question is sanitized
    payload = {
        'question': question.strip(),  # Ensure this field is properly formatted
        'context': context  # Ensure context is passed correctly
    }

    try:
        logger.info(f"Sending request to FastAPI with payload: {payload}")
        response = global_session.post(
            FASTAPI_URL,
            json=payload,
            timeout=10
        )
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Response from FastAPI: {response.json()}")
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        logger.error(f"Response content: {response.content}")
        return {"error": "An HTTP error occurred while calling the FAQ service."}

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        return {"error": "An error occurred while calling the FAQ service."}

    # except requests.exceptions.HTTPError as http_err:
    #     logger.error(f"HTTP error occurred: {http_err}", extra={'action': 'faq_pipeline_http_error'})
    #     return {"error": "An HTTP error occurred while calling the FAQ service."}
    #
    # except requests.exceptions.Timeout:
    #     logger.error("Request to FastAPI timed out", extra={'action': 'faq_pipeline_timeout'})
    #     return {"error": "The request to the FAQ service timed out. Please try again later."}
    #
    # except requests.exceptions.RequestException as req_err:
    #     logger.error(f"Request exception occurred: {req_err}", extra={'action': 'faq_pipeline_error'})
    #     return {"error": "An error occurred while calling the FAQ service."}


@app.route('/videos/<filename>')
def serve_video(filename):
    filename = secure_filename(filename)
    """Serve video files with byte-range support for efficient streaming."""
    range_header = request.headers.get('Range', None)
    video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)

    if not os.path.exists(video_path):
        logger.error(f"File not found: {video_path}", extra={'action': 'serve_video_error', 'file_name': filename})
        return "File not found", 404

    try:
        response = partial_response(video_path, range_header)
        if DEBUG:
            logger.info(f"Serving video file: {filename}", extra={'action': 'serve_video', 'file_name': filename})
        return response
    except Exception as error:
        logger.error(f"Error serving video file: {error}",
                     extra={'action': 'serve_video_exception', 'file_name': filename})
        return "Internal Server Error", 500


def partial_response(file_path, range_header):
    """
    Serve partial content for large video files to support byte-range requests.
    """
    file_size = os.path.getsize(file_path)
    start, end = 0, file_size - 1

    if range_header:
        # Parse the range header to get the start and end bytes
        range_match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            if range_match.group(2):
                end = int(range_match.group(2))

    length = end - start + 1
    with open(file_path, 'rb') as f:
        f.seek(start)
        data = f.read(length)

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Fallback MIME type

    headers = {
        'Content-Range': f'bytes {start}-{end}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(length),
        'Content-Type': mime_type,
    }
    return Response(data, status=206, headers=headers)


@app.route('/fractal', methods=['GET', 'POST'])
def fractal():
    if request.method == 'POST':
        try:
            # Validate and save the uploaded file
            file_path = validate_and_save_file(request)

            # Perform fractal dimension calculation
            fractal_dimension, image_urls, image_file_paths = calculate_fractal_dimension(file_path)

            # Generate PDF report
            pdf_url = generate_report(fractal_dimension, image_file_paths)

            logger.info(f"Fractal dimension calculated: {fractal_dimension}",
                        extra={'action': 'fractal_calculated'})

            # Render the result template with context data
            return render_template(
                'fractal_result.html',
                fractal_dimension=fractal_dimension,
                image_paths=image_urls,
                pdf_url=pdf_url
            )

        except ValueError as error:
            logger.error(f'Error processing image: {error}', extra={'action': 'fractal_error'})
            flash(str(error), 'danger')
            return redirect(url_for('fractal'))

    # For GET requests, render the fractal.html template
    return render_template('fractal.html')


# Utility functions
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type in ['image/png', 'image/jpeg', 'image/gif'] and \
        '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def is_valid_filename(filename):
    """Validate the filename against a specific pattern."""
    # Example pattern: UUID followed by an underscore and then the original filename
    pattern = r'^[a-f0-9\-]{36}_[\w\-]+\.(pdf)$'
    return re.match(pattern, filename) is not None


def validate_and_save_file(requested):
    """Validate the uploaded file and save it to the configured upload folder."""
    if 'file' not in requested.files:
        logger.error('No file part found in the request', extra={'action': 'file_validation_error'})
        raise ValueError('Unable To Validate File')

    file = requested.files['file']
    if file.filename == '':
        logger.error('No file selected', extra={'action': 'file_validation_error'})
        raise ValueError('No selected file')

    if not allowed_file(file.filename):
        logger.error(f"Invalid file format: {file.filename}", extra={'action': 'file_validation_error'})
        raise ValueError('Invalid file format')

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    file_name = f"{unique_id}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
    if not file_path.startswith(upload_folder):
        logger.error(f"Invalid file path attempt: {file_path}", extra={'action': 'file_validation_error'})
        raise ValueError('Invalid file path')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(file_path)
    if DEBUG:
        logger.info(f"File saved successfully at {file_path}",
                    extra={'action': 'file_saved', 'uploaded_filename': file_name})
    return file_path


def preprocess_question(question):
    """Preprocess the user question by stripping whitespaces and ensuring punctuation."""
    question = question.strip()
    if len(question) > 200:
        raise ValueError('Question is too long. Please limit your question to 200 characters.')
    if not question.endswith('?'):
        question += '?'
    if DEBUG:
        logger.info(f"Processed question: {question}", extra={'action': 'question_preprocessed'})
    return question


def safe_redirect(endpoint):
    """Safely redirect to a predefined list of endpoints to avoid Open Redirect vulnerabilities."""
    if endpoint in ALLOWED_REDIRECTS:
        return redirect(url_for(endpoint))
    else:
        logger.warning(f"Invalid redirect attempt to: {endpoint}", extra={'action': 'invalid_redirect_attempt'})
        return redirect(url_for('index'))  # Fallback to home page


def log_response(response):
    """Log outgoing responses with structured data."""
    logger.info({
        'action': 'response_sent',
        'status_code': response.status_code,
        'content_length': response.content_length,
        'method': request.method,
        'url': request.url,
        'remote_addr': request.remote_addr,
        'user_agent': request.user_agent.string,
        'query_string': request.query_string.decode('utf-8'),
    }, extra={'action': 'response_logging'})
    return response


def calculate_fractal_dimension(image_path):
    """Calculate the fractal dimension of an image and save relevant images."""
    try:
        from PIL import Image

        # Ensure the image_path is within the UPLOAD_FOLDER
        if not image_path.startswith(app.config['UPLOAD_FOLDER']):
            logger.error(f"Unauthorized image path: {image_path}",
                         extra={'action': 'fractal_calculation_error'})
            raise ValueError('Invalid image path')

        with Image.open(image_path) as image:
            resized_image, image_gray, image_binary = process_image(image)

        # Perform box counting
        fractal_dimension, log_box_sizes, log_box_counts, intercept = perform_box_counting(image_binary)

        # Delete large variables
        del image_binary

        # Save images and analysis graph using the resized image
        image_urls, image_file_paths = save_images(
            resized_image,
            image_gray,
            None,
            fractal_dimension,
            log_box_sizes,
            log_box_counts,
            intercept
        )
        # Delete more variables
        del image_gray
        del resized_image
        return fractal_dimension, image_urls, image_file_paths
    except Exception as error:
        logger.error(f"Error calculating fractal dimension: {str(error)}",
                     extra={'action': 'fractal_calculation_error'})
        raise


def process_image(image):
    """Resize and convert image to grayscale and binary formats."""
    try:
        if DEBUG:
            logger.info("Converting image to grayscale", extra={'action': 'image_processing'})

        # Resize to a smaller size, e.g., 512x512
        image = image.resize((512, 512))

        # Convert to grayscale if not already
        if image.mode != 'L':
            image_gray_image = image.convert('L')
        else:
            image_gray_image = image.copy()

        # Convert image to NumPy array and normalize
        image_gray = np.array(image_gray_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        if DEBUG:
            logger.info(f"image_gray type: {type(image_gray)}, shape: {image_gray.shape}")

        # Create binary image
        image_binary = image_gray < 0.5
        if DEBUG:
            logger.info(f"image_binary type: {type(image_binary)}, shape: {image_binary.shape}")

        # Return resized image, image_gray, image_binary
        return image, image_gray, image_binary

    except Exception as error:
        logger.error(f"Error processing image: {str(error)}", extra={'action': 'image_processing_error'})
        raise


def perform_box_counting(image_binary):
    """Perform box counting and linear regression to estimate the fractal dimension."""
    try:
        if DEBUG:
            logger.info(f"image_binary type: {type(image_binary)}")
            logger.info(f"image_binary shape: {image_binary.shape}")

        # Proceed with box counting
        # Define box sizes
        min_box_size, max_box_size, n_sizes = 2, min(image_binary.shape) // 4, 10
        sizes = np.floor(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_sizes)).astype(int)
        unique_sizes = np.unique(sizes)
        counts = []

        # Perform box counting
        for size in unique_sizes:
            # Process one size at a time to limit memory usage
            covered_boxes = np.add.reduceat(
                np.add.reduceat(image_binary, np.arange(0, image_binary.shape[0], size), axis=0),
                np.arange(0, image_binary.shape[1], size), axis=1)
            counts.append(np.count_nonzero(covered_boxes > 0))
            del covered_boxes  # Free memory after use

        # Log-transform box sizes and counts
        log_box_sizes, log_box_counts = np.log(unique_sizes), np.log(counts)

        # Perform linear regression on un-centered data
        slope, intercept, r_value, p_value, std_err = linregress(log_box_sizes, log_box_counts)
        fractal_dimension = -slope  # Fractal dimension is the negative slope
        if DEBUG:
            logger.info(f"Fractal dimension calculated: {fractal_dimension}, R-squared: {r_value ** 2}",
                        extra={'action': 'box_counting_done'})
        return fractal_dimension, log_box_sizes, log_box_counts, intercept
    except Exception as error:
        logger.error(f"Error during box counting: {str(error)}", extra={'action': 'box_counting_error'})
        raise


def save_images(image, image_gray, image_binary, fractal_dimension, log_box_sizes, log_box_counts, intercept):
    """Save the images with unique filenames and return their URLs and file paths."""
    try:
        static_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(static_folder, exist_ok=True)

        unique_id = str(uuid.uuid4())

        image_filenames = {
            'original': f'original_{unique_id}.png',
            'grayscale': f'grayscale_{unique_id}.png',
            'analysis': f'analysis_{unique_id}.png'
        }

        image_paths = {key: os.path.join(static_folder, filename) for key, filename in image_filenames.items()}

        # Save images
        image.save(image_paths['original'])
        plt.imsave(image_paths['grayscale'], image_gray, cmap='gray')

        save_fractal_analysis_graph(
            log_box_sizes,
            log_box_counts,
            fractal_dimension,
            intercept,
            image_paths['analysis']
        )

        # Convert file paths to URLs
        image_urls = {
            key: url_for('uploaded_file', filename=filename)
            for key, filename in image_filenames.items()
        }

        # Return both URLs and file paths
        if DEBUG:
            logger.info(f"Images saved successfully: {image_paths}", extra={'action': 'images_saved'})
        return image_urls, image_paths

    except Exception as error:
        logger.error(f"Error saving images: {str(error)}", extra={'action': 'save_images_error'})
        raise


def save_fractal_analysis_graph(log_box_sizes, log_box_counts, fractal_dimension, intercept, plot_path):
    """Generate and save the fractal dimension analysis graph."""
    try:
        plt.figure()

        # Plot actual data points (negating log_box_sizes as before)
        plt.plot(-log_box_sizes, log_box_counts, 'bo', label='Box Counts', markersize=8)
        # Plot fit line using negated log_box_sizes
        fit_line = fractal_dimension * (-log_box_sizes) + intercept
        plt.plot(-log_box_sizes, fit_line, 'r--', label='Fit Line', linewidth=2)
        # Adjust y-axis limits
        plt.ylim([min(log_box_counts) - 1, max(log_box_counts) + 1])

        # Labels and title
        plt.xlabel('Log Box Size')
        plt.ylabel('Log Box Count')
        plt.title('Fractal Dimension Analysis')
        plt.legend()

        # Save the plot with higher resolution
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        if DEBUG:
            logger.info(f"Fractal dimension analysis graph saved to {plot_path}", extra={'action': 'plot_saved'})

    except Exception as error:
        logger.error(f"Error saving fractal dimension analysis graph: {str(error)}",
                     extra={'action': 'save_plot_error'})
        raise


def generate_report(fractal_dimension, image_paths):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        # Generate a unique and secure filename for the PDF report
        pdf_filename = secure_filename(f'fractal_report_{str(uuid.uuid4())}.pdf')
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        c = canvas.Canvas(pdf_path, pagesize=letter)

        # Define page size and layout
        width, height = letter

        # Title of the report
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, height - 0.5 * inch, "Fractal Dimension Analysis Report")

        # Fractal Dimension Text section
        c.setFont("Helvetica", 12)
        fractal_dimension = float(fractal_dimension)
        c.drawString(inch, height - 1 * inch, f"Estimated Fractal Dimension: {float(fractal_dimension):.2f}")

        # Image dimensions for the layout
        image_width, image_height = 3 * inch, 3 * inch  # Fixed size for images

        # Define positions for the images
        positions = [
            (inch, height - 2 * inch - image_height),  # Top-left (Original Image)
            (4.5 * inch, height - 2 * inch - image_height),  # Top-right (Grayscale Image)
            (inch, height - 2 * inch - 2 * image_height - 0.5 * inch),  # Bottom-left (Analysis Graph)
        ]

        labels = ['Original Image', 'Grayscale Image', 'Fractal Analysis']

        # Draw images and labels in the PDF report
        for i, (key, path) in enumerate(image_paths.items()):
            if path:
                image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(path))
                if os.path.exists(image_file_path):
                    x, y = positions[i]
                    c.drawImage(image_file_path, x, y, width=image_width, height=image_height, preserveAspectRatio=True,
                                mask='auto')
                    c.setFont("Helvetica", 10)
                    c.drawCentredString(x + image_width / 2, y - 0.2 * inch, labels[i])
                else:
                    logger.error(f"Image file does not exist: {image_file_path}",
                                 extra={'action': 'missing_image', 'path': image_file_path})
                    raise FileNotFoundError("Image file does not exist.")
            else:
                logger.error(f"Image path is invalid: {path}", extra={'action': 'missing_image', 'path': path})
                raise ValueError("Path does not exist.")

        # Add hyperlink to navigate back to the Projects page
        c.setFont("Helvetica-Bold", 12)
        link_url = url_for('experience', _external=True)  # URL for the projects page
        c.drawString(inch, inch, "Click here to go back to the Projects page")
        c.linkURL(link_url, (inch, inch - 0.2 * inch, 4 * inch, inch + 0.2 * inch))

        # Save the PDF report
        c.save()
        if DEBUG:
            logger.info(f"PDF generated successfully: {pdf_path}", extra={'action': 'pdf_generated'})

        # Convert the PDF file path to a downloadable URL
        pdf_url = url_for('uploaded_file', filename=pdf_filename)

        return pdf_url

    except Exception as error:
        logger.error(f"Error generating PDF: {str(error)}", extra={'action': 'generate_pdf_error'})
        raise


@app.route('/download_generated_report')
@limiter.limit("10 per minute")
def download_generated_report():
    filename = request.args.get('filename')
    if not filename:
        flash('Report not found.', 'danger')
        if DEBUG:
            logger.warning("No filename provided for report download",
                           extra={'action': 'download_report_not_found'})
        return safe_redirect('fractal')

    # Sanitize the filename
    filename = secure_filename(filename)

    # Validate the filename pattern
    if not is_valid_filename(filename):
        flash('Invalid report filename.', 'danger')
        if DEBUG:
            logger.warning(f"Invalid filename pattern: {filename}",
                           extra={'action': 'invalid_filename_pattern'})
        return safe_redirect('fractal')

    # Construct the absolute file path
    upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
    pdf_path = os.path.abspath(os.path.join(upload_folder, filename))

    # Verify that the file is within the UPLOAD_FOLDER to prevent path traversal
    if not pdf_path.startswith(upload_folder):
        flash('Invalid file path.', 'danger')
        if DEBUG:
            logger.warning(f"Invalid file path attempt: {pdf_path}",
                           extra={'action': 'invalid_file_path_attempt'})
        return safe_redirect('fractal')

    # Verify that the file exists
    if not os.path.isfile(pdf_path):
        flash('Report not found.', 'danger')
        if DEBUG:
            logger.warning("Attempted to download a non-existent report",
                           extra={'action': 'download_report_not_found'})
        return safe_redirect('fractal')

    if DEBUG:
        logger.info(f"Report downloaded: {filename}", extra={'action': 'download_report'})
    return send_from_directory(upload_folder, filename, as_attachment=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.warning({
            "message": "No file part in the request"
        })
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning({
            "message": "No selected file"
        })
        return jsonify({"error": "No selected file"}), 400
    if file:
        folder_path = app.config['UPLOAD_FOLDER']
        file_path = os.path.join(folder_path, file.filename)
        try:
            file.save(file_path)
            logger.info({
                "message": f"File saved successfully: {file_path}",
                "file_path": file_path
            })
            return jsonify({"message": "File uploaded successfully"}), 200
        except Exception as error:
            logger.error({
                "message": f"Failed to save file: {file.filename}",
                "error": str(error)
            })
            return jsonify({"error": "Failed to save file"}), 500

#Uncomment the following lines to run the Flask app directly
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render provides the PORT variable; default to 5000 if not set
    app.run(debug=DEBUG, host='0.0.0.0', port=port)
