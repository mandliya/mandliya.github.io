let mermaidTheme = determineComputedTheme();

function initMermaidDiagrams() {
  const codeBlocks = document.querySelectorAll("pre>code.language-mermaid");
  if (codeBlocks.length === 0) return;

  codeBlocks.forEach((elem) => {
    const svgCode = elem.textContent;
    const backup = elem.parentElement;
    backup.classList.add("unloaded");
    let mermaidNode = document.createElement("div");
    mermaidNode.classList.add("mermaid");
    mermaidNode.textContent = svgCode;
    backup.after(mermaidNode);
  });

  mermaid.initialize({ startOnLoad: false, theme: mermaidTheme });
  mermaid.run({ querySelector: ".mermaid" }).then(() => {
    if (typeof d3 !== "undefined") {
      d3.selectAll(".mermaid svg").each(function () {
        var svg = d3.select(this);
        svg.html("<g>" + svg.html() + "</g>");
        var inner = svg.select("g");
        var zoom = d3.zoom().on("zoom", function (event) {
          inner.attr("transform", event.transform);
        });
        svg.call(zoom);
      });
    }
  });
}

if (document.readyState === "complete") {
  initMermaidDiagrams();
} else {
  document.addEventListener("readystatechange", () => {
    if (document.readyState === "complete") {
      initMermaidDiagrams();
    }
  });
}
