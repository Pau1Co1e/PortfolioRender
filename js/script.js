document.addEventListener("DOMContentLoaded", function () {
    // Function to show loading animation
    function showLoading() {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.classList.add('loading-visible');
        }
    }

    // Function to hide loading animation
    function hideLoading() {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.classList.remove('loading-visible');
        }
    }

    // Function to update text color in footer links based on theme
    function updateFooterTextColor() {
        document.querySelectorAll('footer a').forEach(link => {
            const isDarkMode = document.body.classList.contains('dark-mode');
            link.classList.toggle('text-white', isDarkMode);
            link.classList.toggle('text-dark', !isDarkMode);
        });
    }

    // Function to apply dark mode to specified elements
    function applyDarkMode(isDarkMode) {
        const elementsToToggle = [
            document.body,
            document.querySelector('nav'),
            document.querySelector('footer'),
            document.querySelector('.bg-image'),
            document.querySelector('.navbar')
        ];

        elementsToToggle.forEach(element => {
            if (element) {
                element.classList.toggle('dark-mode', isDarkMode);
            }
        });

        document.querySelectorAll('a').forEach(link => link.classList.toggle('dark-mode', isDarkMode));
        document.querySelectorAll('.hero-title').forEach(element => {
            if (element) {
                element.classList.toggle('dark-mode-text', isDarkMode);
            }
        });
        document.querySelectorAll('.text-dark').forEach(element => {
            if (element) {
                element.classList.toggle('text-white', isDarkMode);
                element.classList.toggle('text-dark', !isDarkMode);
            }
        });

        updateFooterTextColor();
    }

    // Function to handle theme toggle
    function handleThemeToggle() {
        showLoading();
        setTimeout(() => {
            const isDarkMode = document.body.classList.toggle('dark-mode');
            applyDarkMode(isDarkMode);
            localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
            hideLoading();
        }, 500);
    }

    // Apply saved theme preference on page load
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        applyDarkMode(true);
    }

    // Event listener for the dark mode toggle button
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', handleThemeToggle);
    }

    // Initial footer text color update
    updateFooterTextColor();
});

document.getElementById('darkModeToggle').addEventListener('click', function() {
    document.body.classList.toggle('dark-mode');
});

document.addEventListener('DOMContentLoaded', function () {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const body = document.body;

    // Check local storage for saved theme preference
    const savedTheme = localStorage.getItem('theme');

    if (savedTheme === 'dark') {
        body.classList.add('dark-mode');
    } else {
        body.classList.remove('dark-mode');
    }

    darkModeToggle.addEventListener('click', function () {
        body.classList.toggle('dark-mode');

        // Save theme preference to local storage
        if (body.classList.contains('dark-mode')) {
            localStorage.setItem('theme', 'dark');
        } else {
            localStorage.setItem('theme', 'light');
        }
    });
});
