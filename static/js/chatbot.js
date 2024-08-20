$(document).ready(function() {
    // Handle form submission for the chatbot
    $('#chatbot-form').on('submit', function(event) {
        event.preventDefault(); // Prevent the form from submitting normally
        const question = $('#question').val();

        if (!question.trim()) {
            alert("Please enter a question.");
            return;
        }

        $.ajax({
            url: '/chatbot-answer',  // Correct the URL to match the Flask route
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ question }),  // Send the question as JSON
            success: function(response) {
                $('#answer').html(`<p>${response.answer}</p>`);  // Display the chatbot's response
            },
            error: function(xhr, status, error) {
                console.error("Status: " + status);
                console.error("Error: " + error);
                console.error("Response Text: " + xhr.responseText);
                alert("An error occurred while processing your request. Please try again.");
            }
        });
    });

    // Handle Dark Mode Toggle
    $('#darkModeToggle').on('click', function() {
        $('body').toggleClass('dark-mode');
        $('.navbar, .footer').toggleClass('dark-mode');
        $('#chatbot-form input, #chatbot-form textarea').toggleClass('dark-mode-input');
    });
});
