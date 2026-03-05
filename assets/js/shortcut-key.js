// Check if the user is on a Mac and update the shortcut key for search accordingly
document.addEventListener("readystatechange", () => {
  if (document.readyState === "interactive") {
    const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
    const shortcutKeyElement = document.querySelector("#search-toggle .nav-link");
    if (shortcutKeyElement && isMac) {
      // use the unicode for command key
      shortcutKeyElement.innerHTML = '&#x2318; k <i class="fa-solid fa-magnifying-glass"></i>';
    }
  }
});

// Open search with Cmd+K (macOS) or Ctrl+K (others).
document.addEventListener("keydown", (event) => {
  const key = (event.key || "").toLowerCase();
  if (key !== "k" || event.altKey || event.shiftKey) {
    return;
  }

  const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
  const usesSearchShortcut = isMac ? event.metaKey : event.ctrlKey;
  if (!usesSearchShortcut) {
    return;
  }

  // Do not override typing shortcuts while focused in editable controls.
  const target = event.target;
  const tagName = target && target.tagName ? target.tagName.toLowerCase() : "";
  const isEditable =
    target &&
    (target.isContentEditable || tagName === "input" || tagName === "textarea" || tagName === "select");
  if (isEditable) {
    return;
  }

  event.preventDefault();
  if (typeof window.openSearchModal === "function") {
    window.openSearchModal();
  }
});
