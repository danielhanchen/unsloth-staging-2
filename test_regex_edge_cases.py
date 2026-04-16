"""Regex edge cases for _find_end_position and _RE patterns."""
from test_helpers import get_fn

_find = get_fn("_find_end_position")


def test_endfor_inside_string_literal():
    """Jinja string containing endfor-like text."""
    t = '{{ "{% endfor %}" }}{% endfor %}'
    r = _find(t)
    assert r is not None
    # Should find the real endfor, not the one in the string
    # (regex can't distinguish, but at least it finds the last one)
    assert r["end"] == len(t)


def test_nested_jinja_comments():
    t = "{# outer {# inner #} still comment #}{% endfor %}"
    r = _find(t)
    assert r is not None


def test_endfor_with_extra_whitespace():
    t = "{%   endfor   %}"
    r = _find(t)
    assert r is not None
    assert "endfor" in r["text"]


def test_endif_with_tabs():
    t = "{%\tendif\t%}"
    r = _find(t)
    assert r is not None


def test_no_space_endfor():
    t = "{%endfor%}"
    r = _find(t)
    assert r is not None


def test_comment_with_newlines_hides_endfor():
    t = "{#\n{% endfor %}\n#}trailing"
    r = _find(t)
    assert r is None


def test_multiple_comments_with_endfor():
    t = "{# {% endfor %} #}{# {% endif %} #}{% endfor %}"
    r = _find(t)
    assert r is not None
    # The real endfor is the last one outside comments
    assert r["start"] > t.rfind("#}")


def test_empty_comment():
    t = "{##}{% endfor %}"
    r = _find(t)
    assert r is not None
