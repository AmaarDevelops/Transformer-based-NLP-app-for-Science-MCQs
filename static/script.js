document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('qa-form');
    const questionInput = document.getElementById('question');
    const submitBtn = document.getElementById('submit-btn');
    const predictionText = document.getElementById('prediction-text');
    const allScoredContainer = document.getElementById('all-scored');
    const resultContainer = document.getElementById('result-container');
    const messageBox = document.getElementById('message-box');
    const messageText = document.getElementById('message-text');

    // The backend URL must match your Python Flask server
    const backendUrl = 'http://127.0.0.1:5000/predict';

    function showMessage(message, isError = false) {
        messageText.textContent = message;
        messageBox.style.color = isError ? 'red' : 'green';
        messageBox.classList.remove('hidden');
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent the default form submission

        showMessage('Predicting...', false);
        resultContainer.classList.add('hidden');

        const question = questionInput.value.trim();
        
        // Collect options from the specific input IDs in your HTML
        const options = [
            document.getElementById('option1').value.trim(),
            document.getElementById('option2').value.trim(),
            document.getElementById('option3').value.trim(),
            document.getElementById('option4').value.trim(),
        ].filter(value => value !== '');

        if (!question || options.length === 0) {
            showMessage('Please enter a question and all four options.', true);
            return;
        }

        submitBtn.disabled = true;

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question, options }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Prediction Failed');
            }

            const finalResult = await response.json();

            // Display the predicted correct option
            const predictedOption = finalResult.predicted_correct_option;
            predictionText.innerHTML = `<strong>${predictedOption}</strong>`;
            
            // Display all predictions in a table
            let tableHTML = `
                <table class="w-full text-left rounded-lg overflow-hidden">
                    <thead class="bg-gray-200">
                        <tr>
                            <th class="p-2">Option</th>
                            <th class="p-2 text-right">Score</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white">
            `;
            
            // Ensure finalResult.all_predictions exists and is an array before iterating
            if (Array.isArray(finalResult.all_predictions)) {
                finalResult.all_predictions.forEach(p => {
                    const isCorrect = p.option === predictedOption;
                    tableHTML += `
                        <tr class="${isCorrect ? 'bg-indigo-100 font-bold' : 'hover:bg-gray-100'}">
                            <td class="p-2">${p.option}</td>
                            <td class="p-2 text-right">${(p.score * 100).toFixed(2)}%</td>
                        </tr>
                    `;
                });
            }

            tableHTML += `
                    </tbody>
                </table>
            `;

            allScoredContainer.innerHTML = tableHTML;
            resultContainer.classList.remove('hidden');
            showMessage('Prediction successful!', false);

        } catch (error) {
            console.error('Fetch error:', error);
            showMessage(`Error: ${error.message}`, true);
            resultContainer.classList.add('hidden');
        } finally {
            submitBtn.disabled = false;
        }
    });
});
