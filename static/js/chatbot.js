$(document).ready(function() {
    var csrf_token = "{{ csrf_token() }}";
    // Set up CSRF token for all AJAX requests
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrf_token);
            }
        }
    });

    // Handle form submission for the chatbot
    $('#chatbot-form').on('submit', function(event) {
        event.preventDefault(); // Prevent the form from submitting normally
        const question = $('#question').val();

        if (!question.trim()) {
            alert("Please enter a question.");
            return;
        }

        $.ajax({
            url: '/chatbot-answer',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ question }),
            success: function(response) {
                $('#answer').html(`<p>${response.answer}</p>`);
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
