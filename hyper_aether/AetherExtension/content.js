console.log("Aether Memory active on this page.");

// Listen for a keyboard shortcut (e.g., Ctrl+Shift+S) to save highlighted text
document.addEventListener('keydown', async (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'S') {
        const selectedText = window.getSelection().toString();
        
        if (selectedText.length > 0) {
            console.log("Sending to Aether OS: ", selectedText);
            
            // Send to your Python backend
            await fetch("http://127.0.0.1:8000/ingest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: selectedText })
            });
            
            alert("Memory saved to Aether OS!");
        }
    }
});