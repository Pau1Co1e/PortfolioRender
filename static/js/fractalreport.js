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
    url: '/.fractalreport',  // Ensure this is pointing to your Netlify function
    type: 'POST',
    data: formData,
    contentType: false,  // Let jQuery handle the content type
    processData: false,  // Don't process the data
    success: function(response) {
        console.log("Raw response:", response);
        const data = typeof response === "string" ? JSON.parse(response) : response;
        console.log("Parsed data:", data);
        if (data.fractalDimension !== undefined) {
            console.log(this.success)
            alert("Fractal Dimension: " + data.fractalDimension);
        } else {
            console.error("Fractal Dimension is undefined");
            alert("Error: Fractal Dimension is undefined.");
        }
    },
    error: function(xhr, status, error) {
        $('#loading-spinner').hide();
        console.error("Status: " + status);
        console.error("Error: " + error);
        console.error("Response Text: " + xhr.responseText);
        alert("Error calculating fractal dimension.");
    }
});
});
