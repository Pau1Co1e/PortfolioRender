import os
import datetime
import logging
import asyncio
import re
import uuid
from flask import (
    Flask, render_template, request, jsonify, flash, redirect, url_for,
    send_file, session, send_from_directory, Response
)
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.utils import secure_filename
from pythonjsonlogger import jsonlogger  # JSON logging
from scipy.stats import linregress
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import torch
from transformers import pipeline

# faq_pipeline = None

# Configure matplotlib
matplotlib.use('Agg')

# Flask app configuration
app = Flask(__name__)

# Constants
ALLOWED_REDIRECTS = {
    'index', 'about_me', 'experience', 'contact', 'download', 'csrf_error',
    'fractal_result', 'fractal', 'chatbot', 'upload', 'financial', 'download_generated_report'
}

# App settings
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config.update(
    UPLOAD_FOLDER=os.path.join(app.root_path, 'static/uploads/'),
    VIDEO_FOLDER=os.path.join(app.root_path, 'static/videos/'),
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
db = SQLAlchemy(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})  # Caching storage with Redis for production

# Logging configuration
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s %(funcName)s %(lineno)d')
logHandler.setFormatter(formatter)
app.logger.addHandler(logHandler)
app.logger.setLevel(logging.INFO)

global faq_pipeline
# Preload the AI model
faq_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=0 if torch.cuda.is_available() else -1
)


# Utility functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_and_save_file(requested):
    """Validate the uploaded file and save it to the configured upload folder."""
    if 'file' not in requested.files:
        app.logger.error('No file part found in the request', extra={'action': 'file_validation_error'})
        raise ValueError('No file part')

    file = requested.files['file']
    if file.filename == '':
        app.logger.error('No file selected', extra={'action': 'file_validation_error'})
        raise ValueError('No selected file')

    if not allowed_file(file.filename):
        app.logger.error(f"Invalid file format: {file.filename}", extra={'action': 'file_validation_error'})
        raise ValueError('Invalid file format')

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(file_path)

    app.logger.info(f"File saved successfully at {file_path}",
                    extra={'action': 'file_saved', 'uploaded_filename': filename})
    return file_path


def preprocess_question(question):
    """Preprocess the user question by stripping whitespaces and ensuring punctuation."""
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
    app.logger.info(f"Processed question: {question}", extra={'action': 'question_preprocessed'})
    return question


def call_faq_pipeline(question, context):
    """Invoke the FAQ pipeline model."""
    # global faq_pipeline
    # from transformers import pipeline
    import torch  # Import torch here
    app.logger.info({
        'action': 'faq_pipeline_called',
        'question': question,
        'context_snippet': context[:100]
    })
    # global faq_pipeline
    # if faq_pipeline is None:
    # faq_pipeline = pipeline(
    #     "question-answering",
    #     model="distilbert-base-cased-distilled-squad",
    #     device=0 if torch.cuda.is_available() else -1
    # )
    # Set model to evaluation mode
    faq_pipeline.model.eval()
    with torch.no_grad():
        result = faq_pipeline(question=question, context=context)
    return result


def safe_redirect(endpoint):
    """Safe redirect to a predefined list of endpoints to avoid open redirect vulnerabilities."""
    if endpoint in ALLOWED_REDIRECTS:
        return redirect(url_for(endpoint))
    else:
        app.logger.warning(f"Invalid redirect to: {endpoint}", extra={'action': 'invalid_redirect_attempt'})
        return redirect(url_for('index'))  # Default to the homepage for safety


# Structured Logging for Requests and Responses
def log_request():
    """Log incoming requests with structured data."""
    app.logger.info({
        'action': 'request_received',
        'method': request.method,
        'url': request.url,
        'remote_addr': request.remote_addr,
        'user_agent': request.user_agent.string,
        'query_string': request.query_string.decode('utf-8'),
    })


def log_response(response):
    """Log outgoing responses with structured data."""
    app.logger.info({
        'action': 'response_sent',
        'status_code': response.status_code,
        'content_length': response.content_length,
    })
    return response


# Flask routes
@app.before_request
def before_request():
    """Actions to perform before each request."""
    log_request()
    db.create_all()


@app.after_request
def after_request(response):
    """Actions to perform after each request."""
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = 'https://portfoliorender-p89i.onrender.com'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return log_response(response)


@app.errorhandler(400)
def bad_request(error):
    app.logger.error(f"Bad Request: {error}", extra={'action': 'bad_request'})
    return f"Bad Request: {error} 400", 400


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    app.logger.error(f"CSRF Error: {e.description}", extra={'action': 'csrf_error'})
    return render_template('csrf_error.html', reason=e.description), 400


@app.route('/')
def index():
    app.logger.info("Rendered index page", extra={'action': 'render_page', 'page': 'index'})
    return render_template('index.html')


@app.route('/about_me')
def about_me():
    app.logger.info("Rendered about_me page", extra={'action': 'render_page', 'page': 'about_me'})
    return render_template('about_me.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Process the form data if provided (you can add your form logic here)
        # Example: Save to database or send email
        flash('Your message has been sent successfully!', 'success')
        app.logger.info('Contact form submitted', extra={'action': 'contact_form_submitted'})
        return redirect(url_for('contact'))

    app.logger.info("Rendered contact page", extra={'action': 'render_page', 'page': 'contact'})
    return render_template('contact.html')


@app.route('/download')
def download():
    app.logger.info("Rendered download page", extra={'action': 'render_page', 'page': 'download'})
    return render_template('download.html')


@app.route('/experience', methods=['GET', 'POST'])
def experience():
    app.logger.info("Rendered experience page", extra={'action': 'render_page', 'page': 'experience'})
    return render_template('experience.html')


@app.route('/chatbot')
def chatbot():
    session['visit_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    app.logger.info(f"Chatbot page accessed at {session['visit_time']}", extra={'action': 'chatbot_accessed'})
    return render_template('chatbot.html')


@app.route('/chatbot-answer', methods=['POST'])
@csrf.exempt
async def chatbot_answer():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            app.logger.warning("No question provided in chatbot request", extra={'action': 'chatbot_error'})
            return jsonify({"error": "No question provided"}), 400

        question = preprocess_question(data['question'])
        cached_response = cache.get(f"chatbot_answer_{question}")
        if cached_response:
            app.logger.info("Returning cached response", extra={'action': 'cached_response_returned'})
            return jsonify(cached_response)

        # Static context
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
            "Other Known Programming Languages: Python, C#, Java, SQL, HTML5/CSS3, JavaScript."
            "My email is engineering.paul@icloud.com."
            "Completed an internship as a robotics software engineer. The internship was my introduction to machine "
            "learning and artificial intelligence and how to apply it in the real world. As an intern, I developed "
            "computer vision, natural language processing, and autonomous functionalities for humanoid robots."

        )

        # Maintain conversation history in the session (limit to last 3 questions)
        session['conversation_history'] = session.get('conversation_history', [])
        session['conversation_history'].append(question)
        # Keep only the last 3 questions
        session['conversation_history'] = session['conversation_history'][-3:]

        # Combine static context with conversation history to create full context
        conversation_context = ' '.join(session['conversation_history'])
        full_context = f"{static_context} {conversation_context}"
        app.logger.info(f"Full context prepared for chatbot", extra={'action': 'context_prepared'})

        # Perform the model inference asynchronously
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, call_faq_pipeline, question, full_context
        )

        answer = result.get('answer', 'Sorry, I could not find an answer.')
        if answer and len(answer) > 3:
            cache.set(f"chatbot_answer_{question}", {"answer": answer}, timeout=60)

        app.logger.info("Chatbot successfully answered question", extra={'action': 'chatbot_answered'})
        return jsonify({"answer": answer})

    except Exception as e:
        app.logger.error(f"Error processing chatbot request: {str(e)}", extra={'action': 'chatbot_error'})
        return jsonify({"error": "An error occurred while processing your question"}), 500


@app.route('/videos/<filename>')
def serve_video(filename):
    """Serve video files with byte-range support for efficient streaming."""
    range_header = request.headers.get('Range', None)
    video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)

    if not os.path.exists(video_path):
        app.logger.error(f"File not found: {video_path}", extra={'action': 'serve_video_error', 'file_name': filename})
        return "File not found", 404

    try:
        response = partial_response(video_path, range_header)
        app.logger.info(f"Serving video file: {filename}", extra={'action': 'serve_video', 'file_name': filename})
        return response
    except Exception as e:
        app.logger.error(f"Error serving video file: {e}",
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

    headers = {
        'Content-Range': f'bytes {start}-{end}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(length),
        'Content-Type': 'video/mp4',
    }

    return Response(data, status=206, headers=headers)


@app.route('/fractal', methods=['GET', 'POST'])
@csrf.exempt
def fractal():
    if request.method == 'POST':
        try:
            # Validate and save the uploaded file
            file_path = validate_and_save_file(request)

            # Perform fractal dimension calculation
            fractal_dimension, image_urls, image_file_paths = calculate_fractal_dimension(file_path)

            # Generate PDF report
            pdf_url = generate_report(fractal_dimension, image_file_paths)

            app.logger.info(f"Fractal dimension calculated: {fractal_dimension}",
                            extra={'action': 'fractal_calculated'})

            # Render the result template with context data
            return render_template(
                'fractal_result.html',
                fractal_dimension=fractal_dimension,
                image_paths=image_urls,
                pdf_url=pdf_url
            )

        except ValueError as e:
            app.logger.error(f'Error processing image: {e}', extra={'action': 'fractal_error'})
            flash(str(e), 'danger')
            return redirect(url_for('fractal'))

    # For GET requests, render the fractal.html template
    return render_template('fractal.html')


def calculate_fractal_dimension(image_path):
    """Calculate the fractal dimension of an image and save relevant images."""
    try:
        from PIL import Image
        with Image.open(image_path) as image:
            resized_image, image_gray, image_binary = process_image(image)
        # Perform box counting
        fractal_dimension, log_box_sizes, log_box_counts, intercept = perform_box_counting(image_binary)
        # Delete large variables
        del image_binary
        # Save images and analysis graph using the resized image
        image_urls, image_file_paths = save_images(resized_image, image_gray, None, fractal_dimension, log_box_sizes,
                                                   log_box_counts, intercept)
        # Delete more variables
        del image_gray
        del resized_image
        return fractal_dimension, image_urls, image_file_paths
    except Exception as e:
        app.logger.error(f"Error calculating fractal dimension: {str(e)}",
                         extra={'action': 'fractal_calculation_error'})
        raise


def process_image(image):
    """Resize and convert image to grayscale and binary formats."""
    try:
        app.logger.info("Converting image to grayscale", extra={'action': 'image_processing'})

        # Resize to a smaller size, e.g., 512x512
        image = image.resize((512, 512))

        # Convert to grayscale if not already
        if image.mode != 'L':
            image_gray_image = image.convert('L')
        else:
            image_gray_image = image.copy()

        # Convert image to NumPy array and normalize
        image_gray = np.array(image_gray_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        app.logger.info(f"image_gray type: {type(image_gray)}, shape: {image_gray.shape}")

        # Create binary image
        image_binary = image_gray < 0.5
        app.logger.info(f"image_binary type: {type(image_binary)}, shape: {image_binary.shape}")

        # Return resized image, image_gray, image_binary
        return image, image_gray, image_binary

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}", extra={'action': 'image_processing_error'})
        raise


def perform_box_counting(image_binary):
    """Perform box counting and linear regression to estimate the fractal dimension."""
    try:
        app.logger.info(f"image_binary type: {type(image_binary)}")
        app.logger.info(f"image_binary shape: {image_binary.shape}")
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
        # Perform linear regression on uncentered data
        slope, intercept, r_value, p_value, std_err = linregress(log_box_sizes, log_box_counts)
        fractal_dimension = -slope  # Fractal dimension is the negative slope
        app.logger.info(f"Fractal dimension calculated: {fractal_dimension}, R-squared: {r_value ** 2}",
                        extra={'action': 'box_counting_done'})
        return fractal_dimension, log_box_sizes, log_box_counts, intercept
    except Exception as e:
        app.logger.error(f"Error during box counting: {str(e)}", extra={'action': 'box_counting_error'})
        raise


def save_images(image, image_gray, image_binary, fractal_dimension, log_box_sizes, log_box_counts, intercept):
    """Save the images with unique filenames and return their URLs and file paths."""
    try:
        from matplotlib import pyplot as plt
        import os

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
        save_fractal_analysis_graph(log_box_sizes, log_box_counts, fractal_dimension, intercept,
                                    image_paths['analysis'])

        # Convert file paths to URLs
        image_urls = {
            key: url_for('uploaded_file', filename=filename)
            for key, filename in image_filenames.items()
        }

        # Return both URLs and file paths
        app.logger.info(f"Images saved successfully: {image_paths}", extra={'action': 'images_saved'})
        return image_urls, image_paths

    except Exception as e:
        app.logger.error(f"Error saving images: {str(e)}", extra={'action': 'save_images_error'})
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
        plt.close()

        app.logger.info(f"Fractal dimension analysis graph saved to {plot_path}", extra={'action': 'plot_saved'})

    except Exception as e:
        app.logger.error(f"Error saving fractal dimension analysis graph: {str(e)}",
                         extra={'action': 'save_plot_error'})
        raise


def generate_report(fractal_dimension, image_paths):
    """Generate a PDF report for the fractal dimension analysis."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
        import os

        pdf_filename = f'fractal_report_{str(uuid.uuid4())}.pdf'
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        c = canvas.Canvas(pdf_path, pagesize=letter)

        # Unpack the page size into width and height
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, height - 0.5 * inch, "Fractal Dimension Analysis Report")

        # Fractal Dimension Text
        c.setFont("Helvetica", 12)
        c.drawString(inch, height - 1 * inch, f"Estimated Fractal Dimension: {fractal_dimension:.2f}")

        # Image dimensions
        image_width, image_height = 3 * inch, 3 * inch  # Fixed size for images

        # Define positions for images
        positions = [
            (inch, height - 2 * inch - image_height),  # Top-left (Original)
            (4.5 * inch, height - 2 * inch - image_height),  # Top-right (Grayscale)
            (inch, height - 2 * inch - 2 * image_height - 0.5 * inch),  # Bottom-left (Analysis)
        ]

        labels = ['Original Image', 'Grayscale Image', 'Fractal Analysis']

        # Draw images and labels
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
                    app.logger.error(f"Image file does not exist: {image_file_path}",
                                     extra={'action': 'missing_image', 'path': image_file_path})
            else:
                app.logger.error(f"Image path is invalid: {path}", extra={'action': 'missing_image', 'path': path})

        # Footer
        c.setFont("Helvetica", 10)
        c.drawString(inch, inch / 2, "Generated by Fractal Dimension Calculator")

        # Save PDF
        c.save()
        app.logger.info(f"PDF generated successfully: {pdf_path}", extra={'action': 'pdf_generated'})

        # Convert file path to URL
        pdf_url = url_for('uploaded_file', filename=pdf_filename)

        return pdf_url

    except Exception as e:
        app.logger.error(f"Error generating PDF: {str(e)}", extra={'action': 'generate_pdf_error'})
        raise


@app.route('/download_generated_report')
def download_generated_report():
    pdf_path = request.args.get('pdf_path')
    if not pdf_path or not os.path.exists(pdf_path):
        flash('Report not found.', 'danger')
        app.logger.warning("Attempted to download a non-existent report", extra={'action': 'download_report_not_found'})
        return safe_redirect('fractal')

    app.logger.info(f"Report downloaded: {pdf_path}", extra={'action': 'download_report'})
    return send_file(pdf_path, as_attachment=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    app.logger.info(f"Serving uploaded file: {filename}",
                    extra={'action': 'serve_uploaded_file', 'file_name': filename})
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def resize_image(image_path, max_size=(1024, 1024)):
    from PIL.Image import UnidentifiedImageError
    """Resize the image to a specified maximum size and return the path."""
    try:
        from PIL import Image
        from PIL.Image import Resampling

        if os.path.isfile(image_path):
            with Image.open(image_path) as img:
                img.thumbnail(max_size, Resampling.LANCZOS)
                resized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + os.path.basename(image_path))
                img.save(resized_path)
                app.logger.info(f"Image resized and saved to {resized_path}",
                                extra={'action': 'image_resized', 'resized_path': resized_path})
                return resized_path
        else:
            app.logger.error(f"File not found for resizing: {image_path}",
                             extra={'action': 'resize_image_error', 'image_path': image_path})
            return None
    except UnidentifiedImageError as e:
        app.logger.error(f"Error resizing image {image_path}: {str(e)}", extra={'action': 'resize_image_error'})
        return None


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render provides the PORT variable; default to 5000 if not set
    app.run(debug=False, host='0.0.0.0', port=port)
