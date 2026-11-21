async function analyzeMessage() {

    const message = document.getElementById("message").value;

    if (message.trim() === "") {
        alert("Please enter a message.");
        return;
    }

    const resultBox = document.getElementById("result");

    resultBox.innerHTML = `
        <h2>Analyzing...</h2>
        <p>Please wait...</p>
    `;

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const result = await response.json();

        if (result.error) {
            resultBox.innerHTML = `
                <h2>Error</h2>
                <p>${result.error}</p>
            `;
            return;
        }

        let confidenceText = "Not available";

        if (result.confidence !== null) {
            confidenceText = result.confidence + "%";
        }

        resultBox.innerHTML = `
            <h2>Analysis Result</h2>

            <p>
                <strong>Detected Intent:</strong>
                ${result.intent}
            </p>

            <p>
                <strong>Confidence:</strong>
                ${confidenceText}
            </p>
        `;

    } catch (error) {

        resultBox.innerHTML = `
            <h2>Connection Error</h2>
            <p>Please make sure the Flask server is running.</p>
        `;
    }
}