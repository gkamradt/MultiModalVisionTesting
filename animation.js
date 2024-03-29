// Assuming your data is stored in a variable named data
// If loading from a file, use d3.json("path/to/results.json").then(function(data) { ... });

d3.json("results_random_tokens.json").then(function(data) {
    updateText(0, data); // Now correctly passing data
});

const container = d3.select("#text-animation");

function updateText(index, data) {
    if (index >= data.length) return;

    const entry = data[index];
    const diff = Diff.diffWords(entry.text_submitted, entry.text_returned);

    // Clear previous text
    container.html("");

    // Add words to the container with appropriate class
    diff.forEach(part => {
        let span = document.createElement("span"); // Create a new span for each part
        if (part.added) {
            span.innerHTML = `<span style="color: red;">${part.value}</span>`; // New word in red
        } else if (part.removed) {
            span.innerHTML = `<span style="text-decoration: line-through;">${part.value}</span>`; // Original word crossed out
        } else {
            span.textContent = part.value + " "; // Unchanged word
        }
        container.node().appendChild(span); // Append the span to the container
    });

    // Wait a bit before showing the next chunk
    setTimeout(() => updateText(index + 1, data), 20);
}