/* Client-side search for the AlphaZero project site.
 *
 * Loads search-index.json lazily on first focus of the search box.
 * Substring + word-prefix matching across page titles, section headings,
 * and section bodies. Dropdown UI with arrow-key navigation.
 */
(function () {
  "use strict";

  let INDEX = null;
  let indexPromise = null;
  let lastQuery = "";
  let selectedIdx = -1;

  function loadIndex() {
    if (INDEX) return Promise.resolve(INDEX);
    if (indexPromise) return indexPromise;
    indexPromise = fetch("./search-index.json", { cache: "force-cache" })
      .then((r) => r.json())
      .then((data) => {
        INDEX = data;
        return data;
      })
      .catch((err) => {
        console.error("search index failed to load", err);
        indexPromise = null;
        return [];
      });
    return indexPromise;
  }

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  function highlightMatch(text, query) {
    if (!query) return escapeHtml(text);
    const re = new RegExp(escapeRegExp(query), "gi");
    const escaped = escapeHtml(text);
    // Re-search the escaped string for the same pattern
    return escaped.replace(re, (m) => `<mark>${m}</mark>`);
  }

  function search(query, index) {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];
    const terms = q.split(/\s+/).filter(Boolean);
    const results = [];

    for (const page of index) {
      const pageTitle = page.title || page.url;
      for (const section of page.sections || []) {
        const heading = section.heading || "";
        const text = section.text || "";
        const hayHeading = heading.toLowerCase();
        const hayText = text.toLowerCase();

        // Score: every term contributes; matches in heading worth more
        let score = 0;
        let firstHitIdx = -1;
        for (const term of terms) {
          let termScore = 0;
          if (hayHeading.includes(term)) {
            termScore += 100;
          }
          const idx = hayText.indexOf(term);
          if (idx >= 0) {
            termScore += 10;
            if (firstHitIdx < 0 || idx < firstHitIdx) firstHitIdx = idx;
          }
          if (termScore === 0) {
            // Require all terms to match at least somewhere
            score = 0;
            break;
          }
          score += termScore;
        }

        if (score === 0) continue;

        // Build excerpt around first text match (or just take heading if no body match)
        let excerpt = "";
        if (firstHitIdx >= 0) {
          const start = Math.max(0, firstHitIdx - 50);
          const end = Math.min(text.length, firstHitIdx + 130);
          excerpt =
            (start > 0 ? "…" : "") +
            text.slice(start, end) +
            (end < text.length ? "…" : "");
        } else {
          excerpt = text.slice(0, 160) + (text.length > 160 ? "…" : "");
        }

        const anchor = section.id ? "#" + section.id : "";
        results.push({
          url: page.url + anchor,
          pageTitle: pageTitle,
          heading: heading || "(no section)",
          excerpt: excerpt,
          score: score,
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, 10);
  }

  function renderResults(query, results, dropdown) {
    if (!results || results.length === 0) {
      if (query.trim().length < 2) {
        dropdown.innerHTML = "";
        dropdown.style.display = "none";
        return;
      }
      dropdown.innerHTML = `<div class="search-empty">No matches for "${escapeHtml(query)}".</div>`;
      dropdown.style.display = "block";
      return;
    }
    const html = results
      .map((r, i) => {
        return `
        <a class="search-result" href="${r.url}" data-idx="${i}">
          <div class="search-result-meta">${highlightMatch(r.pageTitle, query)}<span class="search-result-sep"> · </span><span class="search-result-section">${highlightMatch(r.heading, query)}</span></div>
          <div class="search-result-excerpt">${highlightMatch(r.excerpt, query)}</div>
        </a>`;
      })
      .join("");
    dropdown.innerHTML = html;
    dropdown.style.display = "block";
    selectedIdx = -1;
  }

  function updateSelection(dropdown) {
    const items = dropdown.querySelectorAll(".search-result");
    items.forEach((it, i) => it.classList.toggle("selected", i === selectedIdx));
    if (selectedIdx >= 0 && items[selectedIdx]) {
      items[selectedIdx].scrollIntoView({ block: "nearest" });
    }
  }

  function init() {
    const input = document.getElementById("site-search");
    const dropdown = document.getElementById("site-search-results");
    if (!input || !dropdown) return;

    function performSearch() {
      const q = input.value;
      lastQuery = q;
      if (q.trim().length < 2) {
        dropdown.innerHTML = "";
        dropdown.style.display = "none";
        return;
      }
      loadIndex().then((index) => {
        // Guard against late responses overwriting newer input
        if (lastQuery !== q) return;
        const results = search(q, index);
        renderResults(q, results, dropdown);
      });
    }

    input.addEventListener("input", performSearch);
    input.addEventListener("focus", () => {
      if (input.value.trim().length >= 2) performSearch();
      // Preload index even on empty focus
      loadIndex();
    });

    document.addEventListener("keydown", (e) => {
      const isFocused = document.activeElement === input;
      const dropOpen = dropdown.style.display === "block";

      // Cmd/Ctrl+K opens search anywhere on page
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        input.focus();
        input.select();
        return;
      }

      if (!isFocused && !dropOpen) return;

      if (e.key === "Escape") {
        input.blur();
        dropdown.style.display = "none";
        selectedIdx = -1;
        return;
      }
      if (!dropOpen) return;
      const items = dropdown.querySelectorAll(".search-result");
      if (items.length === 0) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        selectedIdx = Math.min(items.length - 1, selectedIdx + 1);
        updateSelection(dropdown);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        selectedIdx = Math.max(-1, selectedIdx - 1);
        updateSelection(dropdown);
      } else if (e.key === "Enter" && selectedIdx >= 0) {
        e.preventDefault();
        items[selectedIdx].click();
      }
    });

    document.addEventListener("click", (e) => {
      if (!input.contains(e.target) && !dropdown.contains(e.target)) {
        dropdown.style.display = "none";
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
