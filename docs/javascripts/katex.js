document$.subscribe(() => {
    renderMathInElement(document.body, {
        delimiters: [
            { left: "$$", right: "$$", display: true }, // Block math
            { left: "$", right: "$", display: false },  // Inline math
            { left: "\\(", right: "\\)", display: false }, // Inline math (alternative)
            { left: "\\[", right: "\\]", display: true }  // Block math (alternative)
        ],
        throwOnError: false
    });
});