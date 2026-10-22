#!/usr/bin/env python3
"""Answer a whitelisted `gh api` REST GET from GitHub's GraphQL API instead, byte for byte.

    python gh_translate.py --explain -- gh api repos/unslothai/unsloth/pulls/12911 --jq .head.sha
    python gh_translate.py --run -- gh api ...   # prints the answer (exit 0) or exits 3: use REST

REST (core) and GraphQL are separate per-account hourly budgets, so a read that GraphQL can serve
moves load off a spent core budget (the gh shim calls this in GH_GRAPHQL_TRANSLATE=overflow mode,
the default, when core runs low; `always` / `never` force it). A call is translated only when the
answer is PROVABLY the same as the REST one:

  * a plain GET of a whitelisted path (ENDPOINTS below) with known query parameters;
  * a --jq / -q filter that reads only the endpoint's supported fields. The filter is checked by an
    abstract evaluation over the REST schema, and anything it cannot analyse (variables, reduce,
    keys, has, ..., float arithmetic, a whole REST object or array reaching the output) refuses;
  * no --jq at all (a full REST dump), -H / --header, -i / --include, --template, --slurp, -f / -F,
    another host or an unknown flag refuses.

run() then executes the GraphQL query through an injected runner, rebuilds the REST JSON for the
supported fields (REST names and value conventions: lowercase states, numeric ids, `name[bot]`
logins, ghost for deleted users), applies the caller's filter per page like gh does (`--paginate`
runs --jq once per page of per_page items), and prints with gh's encoding (raw strings, compact
JSON with sorted keys and HTML-escaped <>&). Any GraphQL error, truncated nested list or jq failure
returns None, and the caller runs the original REST call unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlsplit

MODES = ("never", "overflow", "always")
PAGINATE_CAP = 1000  # items; a --paginate read past this falls back to REST
NESTED_CAP = 100  # labels per PR, comments per review thread


def mode(env = None):
    v = (
        (env if env is not None else os.environ)
        .get("GH_GRAPHQL_TRANSLATE", "overflow")
        .strip()
        .lower()
    )
    return v if v in MODES else "overflow"


class Unsupported(Exception):
    """This call cannot be proven equal to its REST answer: run the REST call."""


# ------------------------------------------------------------------ REST schemas (supported fields)
S = "scalar"


def arr(t):
    return ("arr", t)


USER = {"login": S, "type": S, "id": S}
LABEL = {"name": S}
PR_LIST_ITEM = {
    "id": S,
    "node_id": S,
    "number": S,
    "title": S,
    "state": S,
    "draft": S,
    "body": S,
    "html_url": S,
    "created_at": S,
    "updated_at": S,
    "closed_at": S,
    "merged_at": S,
    "user": USER,
    "author_association": S,
    "labels": arr(LABEL),
    "head": {"ref": S, "sha": S},
    "base": {"ref": S, "sha": S},
}
PR = dict(
    PR_LIST_ITEM,
    merged = S,
    mergeable = S,
    mergeable_state = S,
    additions = S,
    deletions = S,
    changed_files = S,
    commits = S,
    comments = S,
)
ISSUE = {
    "id": S,
    "node_id": S,
    "number": S,
    "title": S,
    "state": S,
    "state_reason": S,
    "body": S,
    "html_url": S,
    "created_at": S,
    "updated_at": S,
    "closed_at": S,
    "user": USER,
    "author_association": S,
    "labels": arr(LABEL),
    "comments": S,
}
REACTIONS = {
    "total_count": S,
    "+1": S,
    "-1": S,
    "laugh": S,
    "hooray": S,
    "confused": S,
    "heart": S,
    "rocket": S,
    "eyes": S,
}
COMMENT = {
    "id": S,
    "node_id": S,
    "body": S,
    "user": USER,
    "created_at": S,
    "updated_at": S,
    "html_url": S,
    "author_association": S,
    "reactions": REACTIONS,
}
REVIEW = {
    "id": S,
    "node_id": S,
    "body": S,
    "user": USER,
    "state": S,
    "submitted_at": S,
    "commit_id": S,
    "html_url": S,
    "author_association": S,
}
REVIEW_COMMENT = {
    "id": S,
    "node_id": S,
    "body": S,
    "user": USER,
    "path": S,
    "line": S,
    "original_line": S,
    "start_line": S,
    "original_start_line": S,
    "diff_hunk": S,
    "commit_id": S,
    "original_commit_id": S,
    "in_reply_to_id": S,
    "pull_request_review_id": S,
    "created_at": S,
    "updated_at": S,
    "html_url": S,
    "author_association": S,
    "reactions": REACTIONS,
}
# Keys REST leaves out (rather than null) when empty; jq reads a missing key as null, the same.
OPTIONAL = frozenset({"in_reply_to_id"})
FILE = {"filename": S, "status": S, "additions": S, "deletions": S, "changes": S}
STATUS = {
    "state": S,
    "sha": S,
    "total_count": S,
    "statuses": arr({"context": S, "state": S, "description": S, "target_url": S, "created_at": S}),
}
CHECK_RUN = {
    "id": S,
    "node_id": S,
    "name": S,
    "status": S,
    "conclusion": S,
    "started_at": S,
    "completed_at": S,
    "html_url": S,
    "details_url": S,
    "head_sha": S,
    "app": {"slug": S},
}
CHECK_RUNS = {"total_count": S, "check_runs": arr(CHECK_RUN)}
REPO = {
    "id": S,
    "node_id": S,
    "name": S,
    "full_name": S,
    "default_branch": S,
    "private": S,
    "archived": S,
    "fork": S,
    "html_url": S,
    "description": S,
    "owner": {"login": S},
    "visibility": S,
    "stargazers_count": S,
    "forks_count": S,
    "pushed_at": S,
    "created_at": S,
}
BRANCH = {"name": S, "commit": {"sha": S, "commit": {"message": S}}}


# ------------------------------------------------------------------ GraphQL value mapping
_REACT = {
    "THUMBS_UP": "+1",
    "THUMBS_DOWN": "-1",
    "LAUGH": "laugh",
    "HOORAY": "hooray",
    "CONFUSED": "confused",
    "HEART": "heart",
    "ROCKET": "rocket",
    "EYES": "eyes",
}
_FILE_STATUS = {
    "ADDED": "added",
    "DELETED": "removed",
    "MODIFIED": "modified",
    "RENAMED": "renamed",
    "COPIED": "copied",
    "CHANGED": "changed",
}
_MERGEABLE = {"MERGEABLE": True, "CONFLICTING": False, "UNKNOWN": None}
ACTOR = "author{login __typename ... on User{databaseId} ... on Bot{databaseId}}"
REACTION_GROUPS = "reactionGroups{content reactors{totalCount}}"


def _need(d, k):
    """d[k], raising Unsupported when GraphQL left out a field we promised (never guess)."""
    if not isinstance(d, dict) or k not in d:
        raise Unsupported(f"GraphQL answer lacks {k}")
    return d[k]


def _user(a):
    if a is None:  # a deleted account: REST shows the ghost user
        return {"login": "ghost", "type": "User", "id": 10137}
    kind = _need(a, "__typename")
    login = _need(a, "login")
    if kind == "Bot":
        login += "[bot]"
    return {
        "login": login,
        "type": "Bot" if kind == "Bot" else ("User" if kind == "User" else kind),
        "id": a.get("databaseId"),
    }


def _reactions(groups):
    out = {v: 0 for v in _REACT.values()}
    for g in groups or []:
        if g["content"] in _REACT:
            out[_REACT[g["content"]]] = g["reactors"]["totalCount"]
    out["total_count"] = sum(out.values())
    return out


def _nodes(conn, what):
    """A nested connection's nodes, refusing when GraphQL truncated it (REST would show them all)."""
    if (conn.get("pageInfo") or {}).get("hasNextPage"):
        raise Unsupported(f"more than {NESTED_CAP} {what}")
    return conn["nodes"]


# REST key: (GraphQL selection, REST value from the GraphQL node). A call asks only for the keys its
# --jq reads (Plan.keys): check_jq already refuses anything that sees a whole object (length, keys,
# tojson, comparisons, unique, ...), so absent keys cannot change the output, and mergeable /
# mergeStateStatus / body are the slow part of a PR query.
PR_FIELDS = {
    "node_id": ("id", lambda p: p["id"]),
    "id": ("databaseId", lambda p: p["databaseId"]),
    "number": ("number", lambda p: p["number"]),
    "title": ("title", lambda p: p["title"]),
    "state": ("state", lambda p: "open" if p["state"] == "OPEN" else "closed"),
    "draft": ("isDraft", lambda p: p["isDraft"]),
    "body": ("body", lambda p: p["body"] or None),
    "html_url": ("url", lambda p: p["url"]),
    "created_at": ("createdAt", lambda p: p["createdAt"]),
    "updated_at": ("updatedAt", lambda p: p["updatedAt"]),
    "closed_at": ("closedAt", lambda p: p["closedAt"]),
    "merged_at": ("mergedAt", lambda p: p["mergedAt"]),
    "user": (ACTOR, lambda p: _user(p["author"])),
    "author_association": ("authorAssociation", lambda p: p["authorAssociation"]),
    "head": ("headRefName headRefOid", lambda p: {"ref": p["headRefName"], "sha": p["headRefOid"]}),
    "base": ("baseRefName baseRefOid", lambda p: {"ref": p["baseRefName"], "sha": p["baseRefOid"]}),
    "labels": (
        "labels(first:100){nodes{name} pageInfo{hasNextPage}}",
        lambda p: [{"name": n["name"]} for n in _nodes(p["labels"], "labels")],
    ),
    "merged": ("merged", lambda p: p["merged"]),
    "mergeable": ("mergeable", lambda p: _MERGEABLE.get(p["mergeable"])),
    "mergeable_state": ("mergeStateStatus", lambda p: (p["mergeStateStatus"] or "unknown").lower()),
    "additions": ("additions", lambda p: p["additions"]),
    "deletions": ("deletions", lambda p: p["deletions"]),
    "changed_files": ("changedFiles", lambda p: p["changedFiles"]),
    "commits": ("commits{totalCount}", lambda p: p["commits"]["totalCount"]),
    "comments": ("comments{totalCount}", lambda p: p["comments"]["totalCount"]),
}
PR_LIST_KEYS = tuple(PR_LIST_ITEM)
PR_KEYS = tuple(PR)


def _pr_keys(all_keys, wanted):
    """The REST keys to fetch: those the filter reads (all when unknown); never empty, since a
    selection needs a field and `number` is the cheapest."""
    if wanted is None:
        return all_keys
    return tuple(k for k in all_keys if k in wanted) or ("number",)


def _pr_q(keys):
    """Selections in PR_FIELDS order, so the full query is the same string as before projection."""
    return " ".join(sel for k, (sel, _) in PR_FIELDS.items() if k in keys)


def _pr_mapper(keys):
    return lambda p: {k: PR_FIELDS[k][1](p) for k in keys}


PR_LIST_Q = _pr_q(PR_LIST_KEYS)
PR_Q = _pr_q(PR_KEYS)
_pr_list_item = _pr_mapper(PR_LIST_KEYS)
_pr = _pr_mapper(PR_KEYS)


ISSUE_Q = (
    "... on Issue{id databaseId number title state stateReason body url createdAt updatedAt "
    f"closedAt {ACTOR} authorAssociation labels(first:100){{nodes{{name}} pageInfo{{hasNextPage}}}} "
    "comments{totalCount}} __typename"
)


def _issue(i):
    if i.get("__typename") != "Issue":
        raise Unsupported("a pull request: REST adds a pull_request object")
    return {
        "id": i["databaseId"],
        "node_id": i["id"],
        "number": i["number"],
        "title": i["title"],
        "state": i["state"].lower(),
        "state_reason": (i["stateReason"] or "").lower() or None,
        "body": i["body"] or None,
        "html_url": i["url"],
        "created_at": i["createdAt"],
        "updated_at": i["updatedAt"],
        "closed_at": i["closedAt"],
        "user": _user(i["author"]),
        "author_association": i["authorAssociation"],
        "labels": [{"name": n["name"]} for n in _nodes(i["labels"], "labels")],
        "comments": i["comments"]["totalCount"],
    }


COMMENT_Q = (
    f"id databaseId body {ACTOR} createdAt updatedAt url authorAssociation {REACTION_GROUPS}"
)


def _comment(c):
    return {
        "id": c["databaseId"],
        "node_id": c["id"],
        "body": c["body"],
        "user": _user(c["author"]),
        "created_at": c["createdAt"],
        "updated_at": c["updatedAt"],
        "html_url": c["url"],
        "author_association": c["authorAssociation"],
        "reactions": _reactions(c["reactionGroups"]),
    }


REVIEW_Q = f"id databaseId body {ACTOR} state submittedAt commit{{oid}} url authorAssociation"


def _review(r):
    return {
        "id": r["databaseId"],
        "node_id": r["id"],
        "body": r["body"],
        "user": _user(r["author"]),
        "state": r["state"],
        "submitted_at": r["submittedAt"],
        "commit_id": (r["commit"] or {}).get("oid"),
        "html_url": r["url"],
        "author_association": r["authorAssociation"],
    }


REVIEW_COMMENT_Q = (
    f"id databaseId body {ACTOR} path line originalLine startLine originalStartLine "
    "diffHunk commit{oid} originalCommit{oid} replyTo{databaseId} "
    f"pullRequestReview{{databaseId}} createdAt updatedAt url authorAssociation {REACTION_GROUPS}"
)


def _review_comment(c):
    return {
        "id": c["databaseId"],
        "node_id": c["id"],
        "body": c["body"],
        "user": _user(c["author"]),
        "path": c["path"],
        "line": c["line"],
        "original_line": c["originalLine"],
        "start_line": c["startLine"],
        "original_start_line": c["originalStartLine"],
        "diff_hunk": c["diffHunk"],
        "commit_id": (c["commit"] or {}).get("oid"),
        "original_commit_id": (c["originalCommit"] or {}).get("oid"),
        "in_reply_to_id": (c["replyTo"] or {}).get("databaseId"),
        "pull_request_review_id": (c["pullRequestReview"] or {}).get("databaseId"),
        "created_at": c["createdAt"],
        "updated_at": c["updatedAt"],
        "html_url": c["url"],
        "author_association": c["authorAssociation"],
        "reactions": _reactions(c["reactionGroups"]),
    }


def _file(f):
    return {
        "filename": f["path"],
        "status": _FILE_STATUS[f["changeType"]],
        "additions": f["additions"],
        "deletions": f["deletions"],
        "changes": f["additions"] + f["deletions"],
    }


def _repo(r):
    return {
        "id": r["databaseId"],
        "node_id": r["id"],
        "name": r["name"],
        "full_name": r["nameWithOwner"],
        "default_branch": (r["defaultBranchRef"] or {}).get("name"),
        "private": r["isPrivate"],
        "archived": r["isArchived"],
        "fork": r["isFork"],
        "html_url": r["url"],
        "description": r["description"],
        "owner": {"login": r["owner"]["login"]},
        "visibility": r["visibility"].lower(),
        "stargazers_count": r["stargazerCount"],
        "forks_count": r["forkCount"],
        "pushed_at": r["pushedAt"],
        "created_at": r["createdAt"],
    }


def _status(sha, s):
    ctx = s["contexts"] if s else []
    return {
        "state": (s["state"].lower() if s else "pending"),
        "sha": sha,
        "total_count": len(ctx),
        "statuses": [
            {
                "context": c["context"],
                "state": c["state"].lower(),
                "description": c["description"],
                "target_url": c["targetUrl"],
                "created_at": c["createdAt"],
            }
            for c in ctx
        ],
    }


# ------------------------------------------------------------------ the whitelist
@dataclass
class Endpoint:
    name: str
    pattern: re.Pattern
    schema: object  # dict (one object) or ("arr", dict) (a list)
    params: frozenset  # query parameters this endpoint understands
    build: object  # (groups, params) -> (query, variables, list_path or None, mapper)
    paged: bool = False  # a list REST pages with per_page / page
    projects: bool = False  # build() also takes keys=: fetch only the fields the filter reads


def _q(body, variables):
    decl = ", ".join(f"${k}: {t}" for k, (t, _v) in variables.items())
    return (
        f"query({decl}) {{ {body} rateLimit{{cost remaining}} }}"
        if decl
        else f"query {{ {body} rateLimit{{cost remaining}} }}"
    ), {k: v for k, (_t, v) in variables.items()}


def _repo_vars(g, **more):
    return {"o": ("String!", g["o"]), "r": ("String!", g["r"]), **more}


def _b_pr(
    g,
    p,
    keys = None,
):
    keys = _pr_keys(PR_KEYS, keys)
    q, v = _q(
        f"repository(owner:$o, name:$r){{pullRequest(number:$n){{{_pr_q(keys)}}}}}",
        _repo_vars(g, n = ("Int!", int(g["n"]))),
    )
    mapper = _pr_mapper(keys)
    return q, v, None, lambda d: mapper(_need(_need(d, "repository"), "pullRequest") or _missing())


def _missing():
    raise Unsupported("not found: REST returns a 404 body")


def _b_issue(g, p):
    q, v = _q(
        f"repository(owner:$o, name:$r){{issueOrPullRequest(number:$n){{{ISSUE_Q}}}}}",
        _repo_vars(g, n = ("Int!", int(g["n"]))),
    )
    return (
        q,
        v,
        None,
        lambda d: _issue(_need(_need(d, "repository"), "issueOrPullRequest") or _missing()),
    )


def _conn_builder(kind, inner, nodes_q, mapper):
    """A per-PR / per-issue list REST pages chronologically (oldest first)."""

    def build(g, p):
        q, v = _q(
            f"repository(owner:$o, name:$r){{{kind}(number:$n){{{inner}(first:$first, after:$after)"
            f"{{nodes{{{nodes_q}}} pageInfo{{hasNextPage endCursor}}}}}}}}",
            _repo_vars(g, n = ("Int!", int(g["n"])), first = ("Int!", 100), after = ("String", None)),
        )
        return q, v, ("repository", kind, inner), mapper

    return build


def _b_review_comments(g, p):
    """REST lists review comments by id; GraphQL nests them in threads: flatten, then sort."""
    q, v = _q(
        "repository(owner:$o, name:$r){pullRequest(number:$n){reviewThreads(first:$first, after:$after)"
        f"{{nodes{{comments(first:{NESTED_CAP}){{nodes{{{REVIEW_COMMENT_Q}}} pageInfo{{hasNextPage}}}}}} "
        "pageInfo{hasNextPage endCursor}}}}",
        _repo_vars(g, n = ("Int!", int(g["n"])), first = ("Int!", 100), after = ("String", None)),
    )

    def flatten(threads):
        out = [
            _review_comment(c)
            for t in threads
            for c in _nodes(t["comments"], "comments in a thread")
        ]
        return sorted(out, key = lambda c: c["id"])

    return q, v, ("repository", "pullRequest", "reviewThreads"), flatten


def _b_pr_list(
    g,
    p,
    keys = None,
):
    keys = _pr_keys(PR_LIST_KEYS, keys)
    states = {"open": "[OPEN]", "closed": "[CLOSED, MERGED]", "all": "[OPEN, CLOSED, MERGED]"}[
        p.get("state", "open")
    ]
    q, v = _q(
        f"repository(owner:$o, name:$r){{pullRequests(states:{states}, first:$first, after:$after, "
        f"orderBy:{{field:CREATED_AT, direction:DESC}}){{nodes{{{_pr_q(keys)}}} pageInfo{{hasNextPage endCursor}}}}}}",
        _repo_vars(g, first = ("Int!", 100), after = ("String", None)),
    )
    return q, v, ("repository", "pullRequests"), _pr_mapper(keys)


def _b_repo(g, p):
    q, v = _q(
        "repository(owner:$o, name:$r){id databaseId name nameWithOwner defaultBranchRef{name} isPrivate "
        "isArchived isFork url description owner{login} visibility stargazerCount forkCount pushedAt createdAt}",
        _repo_vars(g),
    )
    return q, v, None, lambda d: _repo(_need(d, "repository") or _missing())


def _b_branch(g, p):
    q, v = _q(
        "repository(owner:$o, name:$r){ref(qualifiedName:$ref){name target{oid ... on Commit{message}}}}",
        _repo_vars(g, ref = ("String!", "refs/heads/" + g["b"])),
    )

    def m(d):
        ref = _need(d, "repository")["ref"] or _missing()
        return {
            "name": ref["name"],
            "commit": {
                "sha": ref["target"]["oid"],
                "commit": {"message": ref["target"]["message"]},
            },
        }

    return q, v, None, m


def _b_status(g, p):
    q, v = _q(
        "repository(owner:$o, name:$r){object(expression:$sha){... on Commit{oid status{state "
        "contexts{context state description targetUrl createdAt}}}}}",
        _repo_vars(g, sha = ("String!", g["sha"])),
    )

    def m(d):
        c = _need(d, "repository")["object"] or _missing()
        if len(g["sha"]) != 40 or c["oid"] != g["sha"]:
            raise Unsupported("a ref, not a full sha: REST echoes the resolved sha differently")
        return _status(c["oid"], c["status"])

    return q, v, None, m


def _b_check_runs(g, p):
    """REST lists a commit's latest check runs newest first (by id); GraphQL nests them per suite."""
    if p.get("filter", "latest") != "latest":
        raise Unsupported("check-runs filter=all")
    q, v = _q(
        "repository(owner:$o, name:$r){object(expression:$sha){... on Commit{oid "
        f"checkSuites(first:{NESTED_CAP}){{nodes{{app{{slug}} checkRuns(first:{NESTED_CAP}){{nodes{{id databaseId "
        "name status conclusion startedAt completedAt detailsUrl permalink} pageInfo{hasNextPage}}} "
        "pageInfo{hasNextPage}}}}}",
        _repo_vars(g, sha = ("String!", g["sha"])),
    )
    per_page = int(p.get("per_page", 30))

    def m(d):
        c = _need(d, "repository")["object"] or _missing()
        if c["oid"] != g["sha"]:
            raise Unsupported("a ref, not a full sha")
        runs = [
            {
                "id": r["databaseId"],
                "node_id": r["id"],
                "name": r["name"],
                "status": r["status"].lower(),
                "conclusion": (r["conclusion"] or "").lower() or None,
                "started_at": r["startedAt"],
                "completed_at": r["completedAt"],
                "html_url": r["permalink"],
                "details_url": r["detailsUrl"],
                "head_sha": c["oid"],
                "app": {"slug": (s["app"] or {}).get("slug")},
            }
            for s in _nodes(c["checkSuites"], "check suites")
            for r in _nodes(s["checkRuns"], "check runs")
        ]
        runs.sort(key = lambda r: r["id"], reverse = True)
        return {"total_count": len(runs), "check_runs": runs[:per_page]}

    return q, v, None, m


_OR = r"repos/(?P<o>[A-Za-z0-9_.-]+)/(?P<r>[A-Za-z0-9_.-]+)"
_LIST_PARAMS = frozenset({"per_page", "page"})
ENDPOINTS = [
    Endpoint("pull", re.compile(_OR + r"/pulls/(?P<n>\d+)"), PR, frozenset(), _b_pr, projects = True),
    Endpoint("issue", re.compile(_OR + r"/issues/(?P<n>\d+)"), ISSUE, frozenset(), _b_issue),
    Endpoint(
        "issue comments",
        re.compile(_OR + r"/issues/(?P<n>\d+)/comments"),
        arr(COMMENT),
        _LIST_PARAMS,
        None,
        paged = True,
    ),
    Endpoint(
        "pull reviews",
        re.compile(_OR + r"/pulls/(?P<n>\d+)/reviews"),
        arr(REVIEW),
        _LIST_PARAMS,
        _conn_builder("pullRequest", "reviews", REVIEW_Q, _review),
        paged = True,
    ),
    Endpoint(
        "pull review comments",
        re.compile(_OR + r"/pulls/(?P<n>\d+)/comments"),
        arr(REVIEW_COMMENT),
        _LIST_PARAMS,
        _b_review_comments,
        paged = True,
    ),
    Endpoint(
        "pull files",
        re.compile(_OR + r"/pulls/(?P<n>\d+)/files"),
        arr(FILE),
        _LIST_PARAMS,
        _conn_builder("pullRequest", "files", "path additions deletions changeType", _file),
        paged = True,
    ),
    Endpoint(
        "pull list",
        re.compile(_OR + r"/pulls"),
        arr(PR_LIST_ITEM),
        _LIST_PARAMS | {"state"},
        _b_pr_list,
        paged = True,
        projects = True,
    ),
    Endpoint(
        "commit status",
        re.compile(_OR + r"/commits/(?P<sha>[0-9a-f]{40})/status"),
        STATUS,
        frozenset(),
        _b_status,
    ),
    Endpoint(
        "check runs",
        re.compile(_OR + r"/commits/(?P<sha>[0-9a-f]{40})/check-runs"),
        CHECK_RUNS,
        _LIST_PARAMS | {"filter"},
        _b_check_runs,
    ),
    Endpoint("repo", re.compile(_OR), REPO, frozenset(), _b_repo),
    Endpoint(
        "branch",
        re.compile(_OR + r"/branches/(?P<b>[A-Za-z0-9_./-]+)"),
        BRANCH,
        frozenset(),
        _b_branch,
    ),
]


def _b_issue_comments(g, p):
    """REST serves issues/N/comments for issues and pull requests alike: ask for both shapes."""
    body = (
        "repository(owner:$o, name:$r){issueOrPullRequest(number:$n){__typename "
        f"... on Issue{{comments(first:$first, after:$after){{nodes{{{COMMENT_Q}}} pageInfo{{hasNextPage endCursor}}}}}} "
        f"... on PullRequest{{comments(first:$first, after:$after){{nodes{{{COMMENT_Q}}} pageInfo{{hasNextPage endCursor}}}}}}}}}}"
    )
    q, v = _q(
        body, _repo_vars(g, n = ("Int!", int(g["n"])), first = ("Int!", 100), after = ("String", None))
    )
    return q, v, ("repository", "issueOrPullRequest", "comments"), _comment


ENDPOINTS[2].build = _b_issue_comments


# ------------------------------------------------------------------ jq: abstract evaluation
class T:
    """An abstract jq value: kind scalar | obj | arr; `src` marks values that came from the REST
    document (printing one in full would show fields we do not produce)."""

    __slots__ = ("kind", "fields", "elem", "src")

    def __init__(
        self,
        kind,
        fields = None,
        elem = None,
        src = False,
    ):
        self.kind, self.fields, self.elem, self.src = kind, fields, elem, src


SC = T("scalar")


def from_schema(s):
    if s == S:
        return SC
    if isinstance(s, tuple) and s[0] == "arr":
        return T("arr", elem = from_schema(s[1]), src = True)
    return T("obj", fields = {k: from_schema(v) for k, v in s.items()}, src = True)


_TOKEN = re.compile(
    r"""
    (?P<ws>\s+) |
    (?P<fmt>@[a-z]+) |
    (?P<num>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?) |
    (?P<field>\.[A-Za-z_][A-Za-z0-9_]*) |
    (?P<op>\.\.|==|!=|<=|>=|//|\?//|[.|,()\[\]{}:<>+\-*/%?;]) |
    (?P<ident>\$?[A-Za-z_][A-Za-z0-9_]*) |
    (?P<str>")
""",
    re.X,
)
_FUNCS0 = {
    "length",
    "not",
    "tostring",
    "tojson",
    "ascii_downcase",
    "ascii_upcase",
    "first",
    "last",
    "any",
    "all",
    "sort",
    "unique",
    "min",
    "max",
    "add",
    "type",
    "tonumber",
    "empty",
    "reverse",
    "floor",
    "true",
    "false",
    "null",
    "values",
    "ascii",
}
_FUNCS1 = {
    "map",
    "select",
    "startswith",
    "endswith",
    "contains",
    "test",
    "split",
    "join",
    "ltrimstr",
    "rtrimstr",
    "sort_by",
    "unique_by",
    "group_by",
    "min_by",
    "max_by",
    "first",
    "last",
    "any",
    "all",
    "index",
    "inside",
}
_BANNED = {
    "keys",
    "keys_unsorted",
    "has",
    "to_entries",
    "with_entries",
    "from_entries",
    "del",
    "paths",
    "getpath",
    "path",
    "env",
    "input",
    "inputs",
    "reduce",
    "foreach",
    "def",
    "as",
    "label",
    "limit",
    "try",
    "catch",
    "error",
    "splits",
    "sub",
    "gsub",
    "capture",
    "scan",
    "match",
    "range",
    "tostream",
    "leaf_paths",
    "walk",
    "recurse",
    "debug",
    "stderr",
    "input_line_number",
    "$__loc__",
    "ltrimstr_",
}


def _tokens(src):
    """jq tokens; strings come back as ("str", [parts]) with ("lit", text) / ("expr", tokens)."""
    out, i = [], 0
    while i < len(src):
        m = _TOKEN.match(src, i)
        if not m:
            raise Unsupported(f"jq: cannot read {src[i:i + 12]!r}")
        i = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        if kind == "str":
            parts, buf = [], []
            while True:
                if i >= len(src):
                    raise Unsupported("jq: unterminated string")
                c = src[i]
                if c == '"':
                    i += 1
                    break
                if c == "\\" and src[i + 1 : i + 2] == "(":
                    depth, j = 1, i + 2
                    while j < len(src) and depth:
                        depth += {"(": 1, ")": -1}.get(src[j], 0)
                        j += 1
                    if depth:
                        raise Unsupported("jq: unterminated interpolation")
                    if buf:
                        parts.append(("lit", "".join(buf)))
                        buf = []
                    parts.append(("expr", _tokens(src[i + 2 : j - 1])))
                    i = j
                    continue
                if c == "\\":
                    buf.append(src[i : i + 2])
                    i += 2
                    continue
                buf.append(c)
                i += 1
            if buf:
                parts.append(("lit", "".join(buf)))
            out.append(("str", parts))
            continue
        val = m.group(kind)
        if kind == "num" and not re.fullmatch(r"\d+", val):
            raise Unsupported("jq: non-integer literal (gh formats numbers its own way)")
        if kind == "ident" and (val.startswith("$") or val in _BANNED):
            raise Unsupported(f"jq: {val}")
        if kind == "op" and val in ("..", "?//", ";", "/"):
            raise Unsupported(f"jq: {val}")
        out.append((kind, val))
    return out


class _P:
    """Recursive descent over the token list, evaluating abstract types as it parses."""

    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self, k = 0):
        return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)

    def eat(self, val = None):
        tok = self.peek()
        if tok[0] is None or (val is not None and tok[1] != val):
            raise Unsupported(f"jq: expected {val!r}, got {tok[1]!r}")
        self.i += 1
        return tok

    def full(self, ins):
        out = self.pipe(ins)
        if self.peek()[0] is not None:
            raise Unsupported(f"jq: unexpected {self.peek()[1]!r}")
        return out

    # precedence: | < , < // < or < and < compare < + - < * % < postfix
    def pipe(self, ins):
        out = self.comma(ins)
        while self.peek()[1] == "|":
            self.eat("|")
            out = self.comma(out)
        return out

    def comma(self, ins):
        out = self.alt(ins)
        while self.peek()[1] == ",":
            self.eat(",")
            out = out + self.alt(ins)
        return out

    def alt(self, ins):
        out = self.orx(ins)
        while self.peek()[1] == "//":
            self.eat("//")
            out = out + self.orx(ins)
        return out

    def orx(self, ins):
        out = self.andx(ins)
        while self.peek() == ("ident", "or"):
            self.eat("or")
            self.andx(ins)
            out = [SC]
        return out

    def andx(self, ins):
        out = self.cmp(ins)
        while self.peek() == ("ident", "and"):
            self.eat("and")
            self.cmp(ins)
            out = [SC]
        return out

    def cmp(self, ins):
        out = self.add(ins)
        if self.peek()[1] in ("==", "!=", "<", "<=", ">", ">="):
            self.eat()
            right = self.add(ins)
            for t in out + right:
                if t.kind != "scalar" and t.src:
                    raise Unsupported("jq: comparing a whole REST object or array")
            out = [SC]
        return out

    def add(self, ins):
        out = self.mul(ins)
        while self.peek()[1] in ("+", "-"):
            self.eat()
            right = self.mul(ins)
            for t in out + right:
                if t.kind == "obj":
                    raise Unsupported("jq: object arithmetic")
            out = [x if x.kind == "arr" else SC for x in out]
        return out

    def mul(self, ins):
        out = self.post(ins)
        while self.peek()[1] in ("*", "%"):
            self.eat()
            right = self.post(ins)
            if any(t.kind != "scalar" for t in out + right):
                raise Unsupported("jq: non-scalar arithmetic")
            out = [SC]
        return out

    def post(self, ins):
        if self.peek()[1] == "-":  # unary minus
            self.eat()
            return [SC for _ in self.post(ins)]
        out = self.primary(ins)
        while True:
            tok = self.peek()
            if tok[0] == "field":
                self.eat()
                out = [_field(t, tok[1][1:]) for t in out]
            elif tok[1] == "." and self.peek(1)[0] == "str":
                self.eat()
                key = _lit(self.eat()[1])
                out = [_field(t, key) for t in out]
            elif tok[1] == "[":
                out = self.index(out)
            elif tok[1] == "?":
                self.eat()
            else:
                return out

    def index(self, ins):
        self.eat("[")
        if self.peek()[1] == "]":
            self.eat("]")
            out = []
            for t in ins:
                if t.kind != "arr":
                    raise Unsupported("jq: .[] over an object or scalar")
                out.append(t.elem)
            return out
        tok = self.peek()
        if tok[0] == "str":
            self.eat()
            key = _lit(tok[1])
            self.eat("]")
            return [_field(t, key) for t in ins]
        if tok[1] == "-":
            self.eat()
        if self.eat()[0] != "num":
            raise Unsupported("jq: computed index")
        if self.peek()[1] == ":":
            raise Unsupported("jq: slice")
        self.eat("]")
        out = []
        for t in ins:
            if t.kind != "arr":
                raise Unsupported("jq: indexing a non-array")
            out.append(t.elem)
        return out

    def primary(self, ins):
        tok = self.peek()
        if tok[0] == "field":  # .name at the start of a term
            return ins
        if tok[1] == ".":
            self.eat()
            if self.peek()[0] == "str":  # ."key" (e.g. .reactions."+1")
                key = _lit(self.eat()[1])
                return [_field(t, key) for t in ins]
            return ins  # `.` itself; post() reads a following [...]
        if tok[0] == "num":
            self.eat()
            return [SC]
        if tok[0] == "str":
            self.eat()
            for kind, part in tok[1]:
                if kind == "expr":
                    for t in _P(part).full(ins):
                        _scalarish(t, "string interpolation")
            return [SC]
        if tok[1] == "(":
            self.eat("(")
            out = self.pipe(ins)
            self.eat(")")
            return out
        if tok[1] == "[":
            self.eat("[")
            if self.peek()[1] == "]":
                self.eat("]")
                return [T("arr", elem = SC)]
            items = self.pipe(ins)
            self.eat("]")
            elem = items[0] if items else SC
            for t in items[1:]:
                if t.kind != elem.kind:
                    raise Unsupported("jq: mixed array")
            return [T("arr", elem = elem)]
        if tok[1] == "{":
            return [self.obj(ins)]
        if tok[0] == "fmt":
            self.eat()
            if tok[1] not in ("@tsv", "@csv", "@text", "@json"):
                raise Unsupported(f"jq: {tok[1]}")
            for t in ins:
                if tok[1] in ("@tsv", "@csv"):
                    if t.kind != "arr" or t.elem.kind != "scalar":
                        raise Unsupported(f"jq: {tok[1]} needs an array of scalars")
                else:
                    _scalarish(t, tok[1])
            return [SC]
        if tok == ("ident", "if"):
            return self.ifx(ins)
        if tok[0] == "ident":
            return self.func(ins)
        raise Unsupported(f"jq: {tok[1]!r}")

    def obj(self, ins):
        self.eat("{")
        fields = {}
        while self.peek()[1] != "}":
            k = self.eat()
            if k[0] == "ident":
                key = k[1]
            elif k[0] == "str" and all(p[0] == "lit" for p in k[1]):
                key = _lit(k[1])
            elif k[0] == "field":  # {.x}? not jq; refuse
                raise Unsupported("jq: object key")
            else:
                raise Unsupported("jq: computed object key")
            if self.peek()[1] == ":":
                self.eat(":")
                vals = self.alt(ins)
            else:
                vals = [_field(t, key) for t in ins]
            if len(vals) != 1:
                raise Unsupported("jq: object value with several outputs")
            fields[key] = vals[0]
            if self.peek()[1] == ",":
                self.eat(",")
        self.eat("}")
        return T("obj", fields = fields)

    def ifx(self, ins):
        self.eat("if")
        self.pipe(ins)
        self.eat("then")
        out = self.pipe(ins)
        while self.peek() == ("ident", "elif"):
            self.eat("elif")
            self.pipe(ins)
            self.eat("then")
            out = out + self.pipe(ins)
        if self.peek() == ("ident", "else"):
            self.eat("else")
            out = out + self.pipe(ins)
        else:
            out = out + ins
        self.eat("end")
        return out

    def func(self, ins):
        name = self.eat()[1]
        if name in ("then", "else", "elif", "end", "and", "or"):
            raise Unsupported(f"jq: {name}")
        if self.peek()[1] == "(":
            if name not in _FUNCS1:
                raise Unsupported(f"jq: {name}(...)")
            self.eat("(")
            return self.call1(name, ins)
        if name not in _FUNCS0:
            raise Unsupported(f"jq: {name}")
        if name in ("true", "false", "null"):
            return [SC]
        if name == "empty":
            return []
        out = []
        for t in ins:
            if name == "length":
                if t.kind == "obj":
                    raise Unsupported("jq: length of an object (counts fields)")
                out.append(SC)
            elif name in ("first", "last", "min", "max"):
                if t.kind != "arr":
                    raise Unsupported(f"jq: {name} of a non-array")
                if name in ("min", "max") and t.elem.kind != "scalar":
                    raise Unsupported(f"jq: {name} over objects")
                out.append(t.elem)
            elif name in ("sort", "unique", "reverse"):
                if t.kind != "arr" or t.elem.kind != "scalar":
                    raise Unsupported(f"jq: {name} over objects compares every field")
                out.append(T("arr", elem = SC))
            elif name in ("any", "all", "add"):
                if t.kind != "arr" or t.elem.kind == "obj":
                    raise Unsupported(f"jq: {name}")
                out.append(
                    SC if name != "add" or t.elem.kind == "scalar" else T("arr", elem = t.elem.elem)
                )
            elif name == "values":
                out.append(t)
            else:  # tostring tojson not type ascii_* tonumber floor ascii
                if name in ("tostring", "tojson"):
                    _scalarish(t, name)
                out.append(SC)
        return out

    def _plain_regex(self):
        """test("re"): gh's jq is gojq on Go's RE2, which lacks lookaround and backreferences that
        jq has; only a literal pattern both engines read the same way is translated."""
        tok = self.peek()
        if tok[0] != "str" or not _regex_ok(tok[1]) or self.peek(1)[1] != ")":
            raise Unsupported("jq: test() needs a literal RE2-compatible pattern")

    def call1(self, name, ins):
        out = []
        start = self.i
        for t in ins or [SC]:
            self.i = start
            if name in (
                "map",
                "sort_by",
                "unique_by",
                "group_by",
                "min_by",
                "max_by",
                "any",
                "all",
            ):
                if t.kind != "arr":
                    raise Unsupported(f"jq: {name} over a non-array")
                res = self.pipe([t.elem])
                if name == "map":
                    elem = res[0] if res else SC
                    out.append(T("arr", elem = elem))
                elif name in ("sort_by", "unique_by"):
                    out.append(T("arr", elem = t.elem, src = t.src))
                elif name == "group_by":
                    out.append(T("arr", elem = T("arr", elem = t.elem, src = t.src)))
                elif name in ("min_by", "max_by"):
                    out.append(t.elem)
                else:
                    out.append(SC)
            elif name == "select":
                self.pipe([t])
                out.append(t)
            elif name in ("first", "last"):
                out.extend(self.pipe([t]))
            elif name in ("join",):
                self.pipe([t])
                if t.kind != "arr" or t.elem.kind != "scalar":
                    raise Unsupported("jq: join over objects")
                out.append(SC)
            elif name == "split":
                self.pipe([t])
                out.append(T("arr", elem = SC))
            else:  # startswith endswith contains test ltrimstr rtrimstr index inside
                if name == "test":
                    self._plain_regex()
                for a in self.pipe([t]):
                    _scalarish(a, name)
                if t.kind != "scalar":
                    raise Unsupported(f"jq: {name} on a non-string")
                out.append(SC)
        self.eat(")")
        return out


_RE_DIALECT = re.compile(
    r"\(\?[=!<]|\\[1-9]|\\[kKgG]|[*+?}][+]"
)  # lookaround, backreference, possessive


def _regex_ok(parts):
    return all(k == "lit" for k, _ in parts) and not _RE_DIALECT.search(
        "".join(p for _, p in parts)
    )


def _lit(parts):
    if any(k != "lit" for k, _ in parts):
        raise Unsupported("jq: computed key")
    return json.loads('"' + "".join(p for _, p in parts) + '"')


_READ = None  # check_jq: every field name the filter reads, at any depth


def _field(t, key):
    if _READ is not None:
        _READ.add(key)
    if t.kind == "obj":
        if key not in t.fields:
            raise Unsupported(f"jq: field {key!r} is not translated")
        return t.fields[key]
    raise Unsupported(f"jq: .{key} on a {t.kind}")


def _scalarish(t, why):
    """Values that print the same from GraphQL: scalars and jq-built containers of them."""
    if t.kind == "scalar":
        return
    if t.src:
        raise Unsupported(f"jq: a whole REST {t.kind} reaches {why}")
    for sub in t.fields.values() if t.kind == "obj" else [t.elem]:
        _scalarish(sub, why)


def check_jq(expr, schema):
    """Raise Unsupported unless `expr` only reads supported fields of `schema` and prints values
    the translation reproduces exactly. Returns the field names it reads (every field read goes
    through _field), which is what a projecting endpoint fetches."""
    global _READ
    _READ = read = set()
    try:
        for t in _P(_tokens(expr)).full([from_schema(schema)]):
            _scalarish(t, "the output")
    finally:
        _READ = None
    return frozenset(read)


# ------------------------------------------------------------------ argv -> plan
@dataclass
class Plan:
    endpoint: Endpoint
    groups: dict
    params: dict
    jq: str
    paginate: bool
    per_page: int = 30
    notes: list = field(default_factory = list)
    keys: frozenset | None = None  # field names the filter reads (None: fetch everything)


def _parse_argv(argv):
    a = list(argv)
    if a and os.path.basename(a[0]) == "gh":
        a = a[1:]
    if not a or a[0] != "api":
        raise Unsupported("not `gh api`")
    a = a[1:]
    path, jq, paginate, method = None, None, False, "GET"
    i = 0
    while i < len(a):
        x = a[i]
        val = None
        if x.startswith("--") and "=" in x:
            x, val = x.split("=", 1)
        if x in ("--jq", "-q"):
            jq = val if val is not None else a[i + 1]
            i += 1 if val is not None else 2
        elif x == "--paginate":
            paginate = True
            i += 1
        elif x in ("-X", "--method"):
            method = (val if val is not None else a[i + 1]).upper()
            i += 1 if val is not None else 2
        elif x.startswith("-X") and len(x) > 2:
            method = x[2:].upper()
            i += 1
        elif x == "--cache":
            i += 1 if val is not None else 2
        elif x == "--hostname":
            host = val if val is not None else a[i + 1]
            if host != "github.com":
                raise Unsupported(f"host {host}")
            i += 1 if val is not None else 2
        elif x.startswith("-"):
            raise Unsupported(f"flag {x}")
        elif path is None:
            path = x
            i += 1
        else:
            raise Unsupported(f"extra argument {x!r}")
    if method != "GET":
        raise Unsupported(f"method {method}")
    if not path:
        raise Unsupported("no path")
    return path, jq, paginate


def translate(argv, env = None):
    """A Plan when this `gh api` call can be answered from GraphQL exactly, else None."""
    try:
        return plan_or_raise(argv, env)
    except Unsupported:
        return None


def plan_or_raise(argv, env = None):
    env = os.environ if env is None else env
    if env.get("GH_HOST", "github.com") not in ("github.com", ""):
        raise Unsupported("GH_HOST is not github.com")
    if env.get("GH_REPO") and "{" in " ".join(argv):
        raise Unsupported("{owner}/{repo} placeholders")
    path, jq, paginate = _parse_argv(argv)
    if jq is None:
        raise Unsupported("no --jq: a full REST dump shows fields GraphQL does not translate")
    u = urlsplit(path.lstrip("/"))
    if "{" in u.path:
        raise Unsupported("{owner}/{repo} placeholders")
    params = dict(parse_qsl(u.query, keep_blank_values = True))
    for ep in ENDPOINTS:
        m = ep.pattern.fullmatch(u.path.rstrip("/"))
        if not m:
            continue
        unknown = set(params) - set(ep.params)
        if unknown:
            raise Unsupported(f"query parameter(s) {sorted(unknown)}")
        per_page = 30
        if ep.paged or "per_page" in ep.params:
            try:
                per_page = int(params.get("per_page", 30))
                page = int(params.get("page", 1))
            except ValueError:
                raise Unsupported("non-numeric per_page / page") from None
            if not 1 <= per_page <= 100 or page != 1:
                raise Unsupported("page > 1 or per_page out of range")
        if paginate and not ep.paged and "per_page" in ep.params:
            raise Unsupported("--paginate over an object REST pages (gh runs --jq per page)")
        if ep.name == "pull list" and params.get("state", "open") not in ("open", "closed", "all"):
            raise Unsupported("pull list state")
        keys = check_jq(jq, ep.schema)
        return Plan(ep, m.groupdict(), params, jq, paginate, per_page, keys = keys)
    raise Unsupported(f"path {u.path} is not whitelisted")


# ------------------------------------------------------------------ execution
def _dig(d, path):
    for k in path:
        d = _need(d, k)
        if d is None:
            _missing()
    return d


def default_runner(args):
    """The GraphQL call goes through `gh` again ($GH_TRANSLATE_GH, default the gh on PATH, i.e. the
    shim, so it is cached and paced like any read) with translation off, so it never recurses."""
    env = dict(os.environ, GH_GRAPHQL_TRANSLATE = "never")
    r = subprocess.run(
        [env.get("GH_TRANSLATE_GH") or "gh", *args],
        capture_output = True,
        text = True,
        timeout = 60,
        env = env,
        stdin = subprocess.DEVNULL,
    )
    return r.returncode, r.stdout, r.stderr


def _graphql(runner, query, variables):
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is None:
            continue
        args += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    rc, out, err = runner(args)
    if rc != 0:
        raise Unsupported(f"GraphQL call failed: {err.strip()[:200]}")
    try:
        doc = json.loads(out)
    except ValueError:
        raise Unsupported("GraphQL answer is not JSON") from None
    if doc.get("errors"):
        raise Unsupported(f"GraphQL errors: {str(doc['errors'])[:200]}")
    return doc["data"]


def fetch(plan, runner = default_runner):
    """The REST-shaped document (dict) or list for `plan`, from GraphQL."""
    if plan.endpoint.projects:
        query, variables, list_path, mapper = plan.endpoint.build(
            plan.groups, plan.params, keys = plan.keys
        )
    else:
        query, variables, list_path, mapper = plan.endpoint.build(plan.groups, plan.params)
    if list_path is None:
        return mapper(_graphql(runner, query, variables))
    want = PAGINATE_CAP if plan.paginate else plan.per_page
    variables = dict(
        variables, first = min(100, want if plan.endpoint.name != "pull review comments" else 100)
    )
    items, after = [], None
    while True:
        variables["after"] = after
        conn = _dig(_graphql(runner, query, variables), list_path)
        items.extend(conn["nodes"])
        info = conn["pageInfo"]
        if plan.endpoint.name == "pull review comments":
            if not info["hasNextPage"]:
                break
        elif len(items) >= want or not info["hasNextPage"]:
            break
        if len(items) >= PAGINATE_CAP:
            raise Unsupported(f"more than {PAGINATE_CAP} items")
        after = info["endCursor"]
    if plan.endpoint.name == "pull review comments":
        out = mapper(items)
    else:
        out = [mapper(x) for x in items]
    if not plan.paginate:
        return out[: plan.per_page]
    if len(out) >= PAGINATE_CAP:
        raise Unsupported(f"more than {PAGINATE_CAP} items")
    return out


def gh_encode(v):
    """How gh prints one --jq result: strings raw, everything else compact JSON with sorted keys and
    Go's HTML escaping. A top-level null prints as an empty line (`--jq .mergeable` on a merged PR),
    a nested one as null."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    s = json.dumps(v, separators = (",", ":"), sort_keys = True, ensure_ascii = False)
    return (
        s.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def apply_jq(expr, doc):
    jq = shutil.which("jq")
    if not jq:
        raise Unsupported("jq is not installed")
    r = subprocess.run(
        [jq, "-c", expr], input = json.dumps(doc), capture_output = True, text = True, timeout = 30
    )
    if r.returncode != 0:
        raise Unsupported(f"jq failed: {r.stderr.strip()[:200]}")
    return "".join(
        gh_encode(json.loads(line)) + "\n" for line in r.stdout.splitlines() if line.strip()
    )


def render(plan, doc):
    """gh's stdout for `plan` given the REST-shaped document: --jq per page under --paginate."""
    if plan.paginate and isinstance(doc, list):
        pages = [doc[i : i + plan.per_page] for i in range(0, len(doc), plan.per_page)] or [[]]
        return "".join(apply_jq(plan.jq, p) for p in pages)
    return apply_jq(plan.jq, doc)


def run(plan, runner = default_runner):
    """(stdout, stderr, rc) answered from GraphQL, or None: run the original REST call instead."""
    try:
        return render(plan, fetch(plan, runner)), "", 0
    except (Unsupported, KeyError, TypeError, ValueError, subprocess.TimeoutExpired):
        return None


def explain(argv, env = None):
    try:
        p = plan_or_raise(argv, env)
    except Unsupported as e:
        return f"REST: {e}"
    return (
        f"GraphQL: {p.endpoint.name} {p.groups}"
        + (f", --paginate (up to {PAGINATE_CAP})" if p.paginate else f", first {p.per_page}")
        + f", jq {p.jq!r}"
    )


FALLBACK_RC = 3  # --run: not translated, run the REST call


def main(argv = None, runner = default_runner):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--run"]:
        rest = argv[2:] if argv[1:2] == ["--"] else argv[1:]
        p = translate(rest)
        got = run(p, runner) if p else None
        if got is None:
            return FALLBACK_RC
        sys.stdout.write(got[0])
        return 0
    if argv[:1] == ["--explain"]:
        rest = argv[1:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        print(explain(rest))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
