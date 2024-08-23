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

    $.ajax({
        url: '/fractal_report',
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        headers: {
            'X-CSRFToken': csrfToken  // Include the CSRF token in the headers
        },
        success: function(response) {
            if (response.fractalDimension !== undefined) {
                // Redirect to the results page with parameters
                window.location.href = `/fractal_result?fractal_dimension=${Number(response.fractalDimension).toFixed(2)}&original=${response.image_paths.original}&grayscale=${response.image_paths.grayscale}&binary=${response.image_paths.binary}&analysis=${response.image_paths.analysis}`;
            } else {
                alert("Error: Fractal Dimension is undefined.");
            }
        },
        error: function(xhr, status, error) {
            console.error("Error calculating fractal dimension.");
            alert("Error calculating fractal dimension.");
        }
    });
});
