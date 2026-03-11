let mermaidTheme = determineComputedTheme();

function fixMermaidSvgSizes() {
  const EDGE_PADDING = 8;

  document.querySelectorAll(".mermaid svg").forEach((svg) => {
    const vb = svg.getAttribute("viewBox");
    if (vb) {
      const parts = vb.split(/\s+/).map(Number);
      if (parts.length === 4 && Number.isFinite(parts[2]) && Number.isFinite(parts[3])) {
        let [minX, minY, vbWidth, vbHeight] = parts;
        let maxX = minX + vbWidth;
        let maxY = minY + vbHeight;

        // Some Mermaid diagrams render elements slightly outside the original
        // viewBox. Expand to the actual rendered bounds to prevent clipping/overlap.
        try {
          const bbox = svg.getBBox();
          if (
            bbox &&
            Number.isFinite(bbox.x) &&
            Number.isFinite(bbox.y) &&
            Number.isFinite(bbox.width) &&
            Number.isFinite(bbox.height)
          ) {
            minX = Math.min(minX, bbox.x - EDGE_PADDING);
            minY = Math.min(minY, bbox.y - EDGE_PADDING);
            maxX = Math.max(maxX, bbox.x + bbox.width + EDGE_PADDING);
            maxY = Math.max(maxY, bbox.y + bbox.height + EDGE_PADDING);
          }
        } catch (_err) {
          // Keep original Mermaid sizing if bbox is unavailable.
        }

        const width = Math.max(1, maxX - minX);
        const height = Math.max(1, maxY - minY);

        svg.setAttribute("viewBox", `${minX} ${minY} ${width} ${height}`);
        svg.setAttribute("width", width);
        svg.setAttribute("height", height);
      }
    }

    // Keep diagrams responsive on smaller screens.
    svg.style.maxWidth = "100%";
    svg.style.height = "auto";
    svg.style.display = "block";
    svg.style.margin = "0 auto";
  });
}

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
    fixMermaidSvgSizes();
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

requestAnimationFrame(() => {
  requestAnimationFrame(() => {
    initMermaidDiagrams();
  });
});
