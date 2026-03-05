const getNinjaKeys = () => document.querySelector("ninja-keys");

const applySearchTheme = () => {
  const ninjaKeys = getNinjaKeys();
  if (!ninjaKeys || typeof determineComputedTheme !== "function") {
    return;
  }

  const searchTheme = determineComputedTheme();
  if (searchTheme === "dark") {
    ninjaKeys.classList.add("dark");
  } else {
    ninjaKeys.classList.remove("dark");
  }
};

const openSearchModal = () => {
  const ninjaKeys = getNinjaKeys();
  if (!ninjaKeys) {
    return;
  }

  // Collapse navbar if expanded on mobile.
  const $navbarNav = $("#navbarNav");
  if ($navbarNav.hasClass("show")) {
    $navbarNav.collapse("hide");
  }

  applySearchTheme();
  ninjaKeys.open();
};

// Used by inline onclick and keyboard shortcut code.
window.openSearchModal = openSearchModal;

applySearchTheme();
window.addEventListener("load", applySearchTheme);
