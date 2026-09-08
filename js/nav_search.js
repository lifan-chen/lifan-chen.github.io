(function() {
  function stripHtml(html) {
    return html
      .replace(/<style([\s\S]*?)<\/style>/gi, "")
      .replace(/<script([\s\S]*?)<\/script>/gi, "")
      .replace(/<figure([\s\S]*?)<\/figure>/gi, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function escapeHtml(text) {
    return text.replace(/[&<>"']/g, function(character) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[character];
    });
  }

  function getExcerpt(content, query) {
    var lowerContent = content.toLowerCase();
    var lowerQuery = query.toLowerCase();
    var index = lowerContent.indexOf(lowerQuery);
    var start = index > 30 ? index - 30 : 0;
    var excerpt = content.slice(start, start + 110);
    return (start > 0 ? "..." : "") + excerpt + (start + 110 < content.length ? "..." : "");
  }

  function highlightQuery(text, query) {
    var escaped = escapeHtml(text);
    var escapedQuery = escapeHtml(query);

    if (!escapedQuery) {
      return escaped;
    }

    return escaped.replace(new RegExp(escapedQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), function(match) {
      return "<mark class=\"nav-search-keyword\">" + match + "</mark>";
    });
  }

  $(function() {
    var $form = $("#nav-search-form");
    var $input = $("#nav-search-input");
    var $results = $("#nav-search-results");
    var entries = [];
    var loaded = false;
    var loading = false;

    if (!$form.length || !$input.length || !$results.length) {
      return;
    }

    function loadIndex(callback) {
      if (loaded) {
        callback();
        return;
      }
      if (loading) {
        return;
      }

      loading = true;
      $.ajax({
        url: $form.data("search-path"),
        dataType: "xml",
        success: function(xmlResponse) {
          entries = $("entry", xmlResponse).map(function() {
            return {
              title: $("title", this).text() || "Untitled",
              content: stripHtml($("content", this).text() || ""),
              url: $("link", this).attr("href") || $("url", this).text()
            };
          }).get();
          loaded = true;
          callback();
        },
        complete: function() {
          loading = false;
        }
      });
    }

    function renderResults() {
      var query = $input.val().trim().toLowerCase();

      if (!query) {
        $results.empty().hide();
        return;
      }

      var matches = entries.map(function(entry) {
        var title = entry.title.toLowerCase();
        var content = entry.content.toLowerCase();
        var titleIndex = title.indexOf(query);
        var contentIndex = content.indexOf(query);

        if (titleIndex < 0 && contentIndex < 0) {
          return null;
        }

        return {
          entry: entry,
          rank: titleIndex >= 0 ? 2 : 1
        };
      }).filter(Boolean).sort(function(a, b) {
        return b.rank - a.rank;
      }).slice(0, 6);

      if (!matches.length) {
        $results.html("<div class=\"nav-search-empty\">No results</div>").show();
        return;
      }

      $results.html(matches.map(function(match) {
        var entry = match.entry;
        return [
          "<a class=\"nav-search-result\" role=\"option\" href=\"", escapeHtml(entry.url), "\">",
          "<span class=\"nav-search-title\">", highlightQuery(entry.title, query), "</span>",
          "<span class=\"nav-search-excerpt\">", highlightQuery(getExcerpt(entry.content, query), query), "</span>",
          "</a>"
        ].join("");
      }).join("")).show();
    }

    $input.on("focus input", function() {
      loadIndex(renderResults);
    });

    $form.on("submit", function(event) {
      var firstResult = $results.find("a").first().attr("href");
      event.preventDefault();
      if (firstResult) {
        window.location.href = firstResult;
      }
    });

    $(document).on("click", function(event) {
      if (!$(event.target).closest("#nav-search-form").length) {
        $results.hide();
      }
    });
  });
}());
