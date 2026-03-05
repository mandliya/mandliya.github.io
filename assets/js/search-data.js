---
---

// Build search entries for ninja-keys from site content.
(() => {
  const ninjaKeys = document.querySelector("ninja-keys");
  if (!ninjaKeys) {
    return;
  }

  const data = [];

  {% assign nav_pages = site.pages | where: "nav", true | where_exp: "p", "p.url and p.url != '/'" %}
  {% for p in nav_pages %}
    data.push({
      id: "page-{{ forloop.index0 }}",
      title: {{ p.title | strip_html | strip_newlines | jsonify }},
      section: "Pages",
      handler: () => {
        window.location.href = {{ p.url | relative_url | jsonify }};
      },
    });
  {% endfor %}

  {% for post in site.posts %}
    data.push({
      id: "post-{{ forloop.index0 }}",
      title: {{ post.title | strip_html | strip_newlines | jsonify }},
      description: {{ post.description | default: post.excerpt | strip_html | strip_newlines | truncate: 160 | jsonify }},
      section: "Posts",
      handler: () => {
        window.location.href = {{ post.url | relative_url | jsonify }};
      },
    });
  {% endfor %}

  ninjaKeys.data = data;
})();
