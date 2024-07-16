document.addEventListener('DOMContentLoaded', (event) => {
    const chatbox = document.getElementById('chatbox');
    const imageUrl = '/static/images/chatbox_bg.jpg'; // Correct URL for the image
    chatbox.style.backgroundImage = `url('${imageUrl}')`;
});

function sendPrompt() {
    const prompt = document.getElementById('chatInput').value;
    if (!prompt) return;  // Ensure prompt is not empty

    fetch('/process_query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: prompt })
    })
    .then(response => response.json())
    .then(data => {
        const chatbox = document.getElementById('chatbox');
        const userMessage = document.createElement('div');
        userMessage.classList.add('chat-message', 'user-message');
        userMessage.innerText = prompt;
        chatbox.appendChild(userMessage);

        const botMessage = document.createElement('div');
        botMessage.classList.add('chat-message', 'bot-message');
        botMessage.innerText = Array.isArray(data.response) ? data.response.join("\n") : data.error;
        chatbox.appendChild(botMessage);

        document.getElementById('chatInput').value = '';  // Clear the input field
        chatbox.scrollTop = chatbox.scrollHeight;  // Scroll to the bottom
    })
    .catch(error => console.error('Error:', error));
}

function addDataset() {
    const fileInput = document.getElementById('file');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/add_client_dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('uploadResult').innerText = data.message;
    })
    .catch(error => console.error('Error:', error));
}

function downloadChatAsPDF() {
    const { jsPDF } = window.jspdf;
    const chatbox = document.getElementById('chatbox');
    const messages = chatbox.innerText;
    const doc = new jsPDF();
    doc.text(messages, 10, 10);
    doc.save('chat_history.pdf');
}
