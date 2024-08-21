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

    $.ajax({
        url: '/fractalreport',
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        success: function(response) {
            if (response.fractalDimension !== undefined) {
                // Redirect to the results page with parameters
                window.location.href = `/fractal_result?fractal_dimension=${response.fractalDimension}&original=${response.image_paths.original}&grayscale=${response.image_paths.grayscale}&binary=${response.image_paths.binary}&analysis=${response.image_paths.analysis}`;
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
