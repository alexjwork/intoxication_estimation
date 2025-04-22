document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('#upload-form');
    if (form) {
        form.addEventListener('submit', (e) => {
            const image = document.getElementById('image').files[0];
            const audio = document.getElementById('audio').files[0];

            // Validate file presence
            if (!image || !audio) {
                e.preventDefault();
                alert('Please upload both an image and an audio file.');
                return;
            }

            // Validate file types
            const validImageTypes = ['image/png', 'image/jpeg'];
            const validAudioTypes = ['audio/wav', 'audio/mpeg'];
            if (!validImageTypes.includes(image.type)) {
                e.preventDefault();
                alert('Image must be PNG or JPEG.');
                return;
            }
            if (!validAudioTypes.includes(audio.type)) {
                e.preventDefault();
                alert('Audio must be WAV or MP3.');
                return;
            }

            // Validate file sizes (max 5MB)
            const maxSize = 5 * 1024 * 1024; // 5MB
            if (image.size > maxSize || audio.size > maxSize) {
                e.preventDefault();
                alert('Files must be smaller than 5MB.');
                return;
            }
        });
    }
});
