document.addEventListener('DOMContentLoaded', function() {
    // Ensure jQuery is loaded before using $
    if (typeof $ === 'undefined') {
        console.error('jQuery is not loaded');
        return;
    }

    // Attach the event listener for form submission
    $('form').on('submit', function(event) {
        event.preventDefault();  // Prevent the form from submitting normally

        const fileInput = document.getElementById('file');
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        // Get the CSRF token from the form
        const csrfToken = $('input[name="csrf_token"]').val();
        formData.append('csrf_token', csrfToken);

        // Perform the AJAX request
        $.ajax({
            type: 'POST',
            url: '/fractal_result',
            data: formData,
            contentType: false,  // Prevent jQuery from overriding the `contentType`
            processData: false,  // Prevent jQuery from processing the FormData object
            headers: {
                'X-CSRFToken': csrfToken // Include the CSRF token in the headers
            },
            success: function(response) {
                if (response.fractalDimension !== undefined) {
                    // Redirect to the results page with query parameters
                    const fractalDimension = Number(response.fractalDimension).toFixed(2);
                    window.location.href = `/fractal_result?fractal_dimension=${fractalDimension}&original=${response.image_paths.original}&grayscale=${response.image_paths.grayscale}&binary=${response.image_paths.binary}&analysis=${response.image_paths.analysis}`;
                } else {
                    alert("Error: Fractal Dimension is undefined.");
                }
            },
            error: function(xhr, status, ) {
                console.error("Error calculating fractal dimension. ", xhr.responseText, status, error);
                alert("Error calculating fractal dimension.");
            }
        });
    });
});
