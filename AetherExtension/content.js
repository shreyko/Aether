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
// Listen for a keyboard shortcut (Ctrl+Shift+E) to RETRIEVE memory
document.addEventListener('keydown', async (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'E') {
        
        // 1. Try to get highlighted text first (Most reliable method)
        let queryText = window.getSelection().toString();
        
        // 2. If nothing is highlighted, try getting the text from the active input box
        if (!queryText || queryText.trim().length === 0) {
            let activeEl = document.activeElement;
            if (activeEl.tagName === "TEXTAREA" || activeEl.tagName === "INPUT") {
                queryText = activeEl.value;
            } else if (activeEl.isContentEditable) {
                queryText = activeEl.innerText;
            }
        }
        
        if (queryText && queryText.trim().length > 0) {
            console.log("Asking Aether OS for context about: ", queryText);
            
            try {
                let response = await fetch("http://127.0.0.1:8000/retrieve", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: queryText })
                });
                
                let data = await response.json();
                
                // Check if context exists and is not empty
                if (data.context && data.context.length > 0) {
                    
                    // Handle both Array and String responses
                    let memoryText = Array.isArray(data.context) ? data.context.join("\n") : data.context;
                    
                    let contextString = "\n\n[Aether Memory Context]:\n" + memoryText;
                    let fullText = queryText + contextString;
                    
                    // Try copying to clipboard
                    await navigator.clipboard.writeText(fullText);
                    alert("Aether Context found! Press Ctrl+V to paste it into the chat.");
                } else {
                    alert("Aether OS: No relevant memories found.");
                }
            } catch (err) {
                alert("Extension Crash: " + err.message);
                console.error(err);
            }
        } else {
            alert("Please HIGHLIGHT your question first, then press Ctrl+Shift+E.");
        }
    }
});