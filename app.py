import os
import datetime
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, CSRFError
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
db = SQLAlchemy(app)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


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
    return "Bad Request!", 400


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
        # Lazy import for memory efficiency
        from transformers import pipeline

        data = request.get_json()
        logger.info(f"Received data: {data}")

        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        faq_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device="cpu")

        context = (
            "My name is Paul Coleman. I am an AI and ML Engineer focused on building innovative solutions in "
            "Artificial Intelligence and Machine Learning. Feel free to ask about my projects, experience, or anything AI/ML related."
        )
        result = faq_pipeline(question=data['question'], context=context)
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
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def calculate_fractal_dimension(image_path):
    """Calculate the fractal dimension of an image and save relevant images."""
    try:
        # Lazy import for memory efficiency
        from PIL import Image
        from skimage.color import rgb2gray
        import numpy as np

        with Image.open(image_path) as image:
            image_gray, image_binary = process_image(image)

        # Perform box counting
        fractal_dimension, log_box_sizes, log_box_counts, intercept = perform_box_counting(image_binary)

        # Save images and analysis graph
        image_paths = save_images(image, image_gray, image_binary, fractal_dimension, log_box_sizes, log_box_counts, intercept)

        return fractal_dimension, image_paths

    except Exception as e:
        logger.error(f"Error calculating fractal dimension: {str(e)}")
        raise


def process_image(image):
    """Convert image to grayscale and binary formats."""
    try:
        logger.info("Converting image to grayscale")
        from skimage.color import rgb2gray
        import numpy as np

        image = image.resize((1024, 1024))  # Resize image to a manageable size
        image_gray = rgb2gray(np.array(image))
        logger.info("Converting image to binary")
        image_binary = image_gray < 0.5
        return image_gray, image_binary
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def perform_box_counting(image_binary):
    """Perform box counting to estimate the fractal dimension."""
    try:
        from scipy.stats import linregress
        import numpy as np

        # Box sizes and counts for box counting
        min_box_size, max_box_size, n_sizes = 2, min(image_binary.shape) // 4, 10
        box_sizes, box_counts = box_count(image_binary, min_box_size, max_box_size, n_sizes)

        # Calculate log values
        log_box_sizes, log_box_counts = np.log(box_sizes), np.log(box_counts)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_box_sizes, log_box_counts)
        fractal_dimension = -slope

        logger.info(f"Fractal dimension calculated: {fractal_dimension}, R-squared: {r_value ** 2}")
        return fractal_dimension, log_box_sizes, log_box_counts, intercept

    except Exception as e:
        logger.error(f"Error during box counting: {str(e)}")
        raise

def box_count(img, min_box_size, max_box_size, n_sizes):
    """Box counting algorithm to compute the number of boxes containing parts of the image."""
    try:
        import numpy as np

        sizes = np.floor(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=n_sizes)).astype(int)
        unique_sizes = np.unique(sizes)
        counts = []

        for size in unique_sizes:
            covered_boxes = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
                                            np.arange(0, img.shape[1], size), axis=1)
            counts.append(np.count_nonzero(covered_boxes > 0))

        return unique_sizes, counts
    except Exception as e:
        logger.error(f"Error in box_count function: {str(e)}")
        raise


def save_images(image, image_gray, image_binary, fractal_dimension, log_box_sizes, log_box_counts, intercept):
    """Save the original, grayscale, binary images, and the fractal dimension analysis graph."""
    try:
        from matplotlib import pyplot as plt
        import os

        # Ensure the static folder exists
        static_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(static_folder, exist_ok=True)

        image_paths = {
            'original': os.path.join(static_folder, 'original.png'),
            'grayscale': os.path.join(static_folder, 'grayscale.png'),
            'binary': os.path.join(static_folder, 'binary.png'),
            'analysis': os.path.join(static_folder, 'analysis.png')
        }

        image.save(image_paths['original'])
        plt.imsave(image_paths['grayscale'], image_gray, cmap='gray')
        plt.imsave(image_paths['binary'], image_binary, cmap='binary')

        # Save the fractal analysis graph
        save_fractal_analysis_graph(log_box_sizes, log_box_counts, fractal_dimension, intercept,
                                    image_paths['analysis'])

        logger.info(f"Images saved successfully: {image_paths}")
        return image_paths

    except Exception as e:
        logger.error(f"Error saving images: {str(e)}")
        raise


def save_fractal_analysis_graph(log_box_sizes, log_box_counts, fractal_dimension, intercept, plot_path):
    """Generate and save the fractal dimension analysis graph."""
    try:
        from matplotlib import pyplot as plt
        import numpy as np

        plt.figure()

        # Direct calculation of the fit line using the calculated slope and intercept
        fit_line = intercept + fractal_dimension * log_box_sizes

        # Check if the slope has the correct sign
        if fractal_dimension > 0:
            fractal_dimension = -fractal_dimension
            fit_line = intercept + fractal_dimension * log_box_sizes

        # Debugging: Print fit line values
        print("Log Box Sizes:", log_box_sizes)
        print("Log Box Counts:", log_box_counts)
        print("Fit Line Values:", fit_line)

        # Plot the actual data points (box counts)
        plt.plot(log_box_sizes, log_box_counts, 'bo', label='Box Counts', markersize=8)

        # Plot the fit line
        plt.plot(log_box_sizes, fit_line, 'r--', label='Fit Line', linewidth=2)

        # Adjust y-axis limits to ensure both the data points and the line are visible
        plt.ylim([min(log_box_counts) - 1, max(log_box_counts) + 1])

        # Set labels and title
        plt.xlabel('Log Box Size')
        plt.ylabel('Log Box Count')
        plt.title('Fractal Dimension Analysis')
        plt.legend()

        # Save the plot with higher resolution
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Fractal dimension analysis graph saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error saving fractal dimension analysis graph: {str(e)}")
        raise


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)


def resize_image(image_path, max_size=(1024, 1024)):
    """Resize the image to a specified maximum size and return the path."""
    try:
        from PIL import Image, UnidentifiedImageError
        from PIL.Image import Resampling
        import os

        if os.path.isfile(image_path):
            with Image.open(image_path) as img:
                img.thumbnail(max_size, Resampling.LANCZOS)
                resized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + os.path.basename(image_path))
                img.save(resized_path)
                return resized_path
        else:
            logger.error(f"File not found for resizing: {image_path}")
            return None
    except UnidentifiedImageError as e:
        logger.error(f"Error resizing image {image_path}: {str(e)}")
        return None


def generate_report(fractal_dimension, image_paths):
    """Generate a PDF report for the fractal dimension analysis."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
        import os

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fractal_report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
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
            (inch, height - 2 * inch - 2 * image_height - 0.5 * inch),  # Bottom-left (Binary)
            (4.5 * inch, height - 2 * inch - 2 * image_height - 0.5 * inch)  # Bottom-right (Analysis)
        ]

        labels = ['Original Image', 'Grayscale Image', 'Binary Image', 'Fractal Dimension Analysis']

        # Draw images and labels
        for i, (key, path) in enumerate(image_paths.items()):
            if path and os.path.exists(path):
                x, y = positions[i]
                c.drawImage(path, x, y, width=image_width, height=image_height, preserveAspectRatio=True, mask='auto')
                c.setFont("Helvetica", 10)
                c.drawCentredString(x + image_width / 2, y - 0.2 * inch, labels[i])  # Label below each image
            else:
                logger.error(f"Image path is invalid or file does not exist: {path}")

        # Footer
        c.setFont("Helvetica", 10)
        c.drawString(inch, inch / 2, "Generated by Fractal Dimension Calculator")

        # Save PDF
        c.save()
        logger.info(f"PDF generated successfully: {pdf_path}")
        return pdf_path

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise


@app.route('/download_generated_report')
def download_generated_report():
    pdf_path = request.args.get('pdf_path')
    if not pdf_path or not os.path.exists(pdf_path):
        flash('Report not found.', 'danger')
        return safe_redirect('fractal')

    return send_file(pdf_path, as_attachment=True)


def safe_redirect(endpoint):
    """Safe redirect to a predefined list of endpoints to avoid open redirect vulnerabilities."""
    if endpoint in ALLOWED_REDIRECTS:
        return redirect(url_for(ALLOWED_REDIRECTS[endpoint]))
    else:
        return redirect(url_for('index'))  # Default to the homepage for safety


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render provides the PORT variable; default to 5000 if not set
    app.run(debug=True, host='0.0.0.0', port=port)
