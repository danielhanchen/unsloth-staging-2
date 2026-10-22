#!/usr/bin/env python3
"""GitHub reads through GraphQL, returned in the REST shapes our scripts already parse.

GraphQL has its own 5000 points/hour bucket per account, separate from REST core, so moving the
polling reads here spreads load across both buckets (it is not free: a query costs about one point
per 100 nodes requested). One query also replaces several REST round trips: `pr_lists` fetches a
PR's reviews, issue comments (with who reacted), review comments and commits at once, where the
REST path is one paginated call per list plus one per comment's reactions.

    import gh_graphql as gq
    data = gq.query("query($o:String!,$n:String!){repository(owner:$o,name:$n){id}}", o="a", n="b")
    lists = gq.pr_lists("unslothai/unsloth", 123)   # {"repos/.../pulls/123/reviews": [...], ...}

A connection with more pages than one query fetches is left out of `pr_lists`, so the caller reads
it over REST as before; any GraphQL error raises GraphQLError for the caller to fall back on.
GH_GRAPHQL=0 turns the GraphQL paths off; GH_GRAPHQL_DEBUG=1 logs each query's cost and the
points left (`rateLimit` is accurate, unlike `gh api rate_limit`).

GitHub Actions (runs, jobs, logs, artifacts, check runs) and PR file patches are REST-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


class GraphQLError(RuntimeError):
    """gh failed, the response was not JSON, or GraphQL returned errors."""


def enabled():
    return os.environ.get("GH_GRAPHQL", "1") != "0"


def query(
    q,
    timeout = 120,
    env = None,
    **variables,
):
    """`data` of one GraphQL query run through `gh api graphql` (strings via -f, ints via -F)."""
    args = ["gh", "api", "graphql", "-f", f"query={q}"]
    for k, v in variables.items():
        args += (
            ["-F", f"{k}={v}"]
            if isinstance(v, int) and not isinstance(v, bool)
            else ["-f", f"{k}={v}"]
        )
    try:
        r = subprocess.run(args, capture_output = True, text = True, timeout = timeout, env = env)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GraphQLError(f"gh api graphql: {e}") from e
    try:
        out = json.loads(r.stdout or "{}")
    except json.JSONDecodeError as e:
        raise GraphQLError(
            f"gh api graphql rc={r.returncode}: {(r.stderr or r.stdout)[:200]}"
        ) from e
    if r.returncode != 0 or out.get("errors") or "data" not in out:
        msg = (
            "; ".join(e.get("message", "") for e in out.get("errors") or [])
            or (r.stderr or "")[:200]
        )
        raise GraphQLError(f"gh api graphql rc={r.returncode}: {msg}")
    rl = (out["data"] or {}).get("rateLimit")
    if rl and os.environ.get("GH_GRAPHQL_DEBUG"):
        print(
            f"gh_graphql: cost {rl.get('cost')}, {rl.get('remaining')} left, resets {rl.get('resetAt')}",
            file = sys.stderr,
        )
    return out["data"]


# ----------------------------------------------------------------- REST-shaped adapters
_REACTION = {
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


def rest_user(actor):
    """GraphQL actor -> REST `user`: a Bot's REST login carries the `[bot]` suffix."""
    if not actor:
        return None
    login = actor.get("login") or ""
    bot = actor.get("__typename") == "Bot"
    return {
        "login": f"{login}[bot]" if bot and not login.endswith("[bot]") else login,
        "type": "Bot" if bot else "User",
    }


def rest_reactions(groups):
    """reactionGroups -> REST's per-item `reactions` summary ({"+1": n, ..., "total_count": n})."""
    out = {v: 0 for v in _REACTION.values()}
    for g in groups or []:
        k = _REACTION.get(g.get("content"))
        if k:
            out[k] = ((g.get("reactors") or {}).get("totalCount")) or 0
    out["total_count"] = sum(out.values())
    return out


def rest_reaction_list(groups):
    """reactionGroups (with reactor nodes) -> REST `issues/comments/ID/reactions`, or None when
    a group has more reactors than were fetched."""
    out = []
    for g in groups or []:
        k = _REACTION.get(g.get("content"))
        rs = g.get("reactors") or {}
        nodes = rs.get("nodes") or []
        if (rs.get("totalCount") or 0) > len(nodes):
            return None
        out += [{"content": k, "user": rest_user(a)} for a in nodes if k]
    return out


_ACTOR = "author { __typename login }"
_REACTORS = "reactionGroups { content reactors(first: 10) { totalCount nodes { __typename ... on Actor { login } } } }"
_COUNTS = "reactionGroups { content reactors { totalCount } }"
PAGE = 100

PR_LISTS_Q = f"""
query($owner: String!, $name: String!, $n: Int!) {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: $owner, name: $name) {{
    pullRequest(number: $n) {{
      reviews(first: {PAGE}) {{ pageInfo {{ hasNextPage }}
        nodes {{ id databaseId {_ACTOR} state submittedAt body url commit {{ oid }} }} }}
      comments(first: {PAGE}) {{ pageInfo {{ hasNextPage }}
        nodes {{ id databaseId {_ACTOR} body createdAt updatedAt url {_REACTORS} }} }}
      reviewThreads(first: {PAGE}) {{ pageInfo {{ hasNextPage }}
        nodes {{ comments(first: 50) {{ pageInfo {{ hasNextPage }}
          nodes {{ id databaseId {_ACTOR} body path line originalLine outdated createdAt updatedAt url
                  diffHunk commit {{ oid }} replyTo {{ databaseId }} pullRequestReview {{ databaseId }}
                  {_COUNTS} }} }} }} }}
      commits(first: {PAGE}) {{ pageInfo {{ hasNextPage }}
        nodes {{ commit {{ oid message authoredDate committedDate
          author {{ name email date user {{ login }} }} committer {{ name email date }} }} }} }}
    }}
  }}
}}"""


def _review(r):
    return {
        "id": r.get("databaseId"),
        "node_id": r.get("id"),
        "user": rest_user(r.get("author")),
        "state": r.get("state"),
        "submitted_at": r.get("submittedAt"),
        "body": r.get("body") or "",
        "html_url": r.get("url"),
        "commit_id": (r.get("commit") or {}).get("oid"),
    }


def _issue_comment(c):
    return {
        "id": c.get("databaseId"),
        "node_id": c.get("id"),
        "user": rest_user(c.get("author")),
        "body": c.get("body") or "",
        "created_at": c.get("createdAt"),
        "updated_at": c.get("updatedAt"),
        "html_url": c.get("url"),
        "reactions": rest_reactions(c.get("reactionGroups")),
    }


def _review_comment(c):
    outdated = bool(c.get("outdated"))
    return {
        "id": c.get("databaseId"),
        "node_id": c.get("id"),
        "user": rest_user(c.get("author")),
        "body": c.get("body") or "",
        "path": c.get("path"),
        "line": c.get("line"),
        "original_line": c.get("originalLine"),
        # REST nulls `position` once the anchored hunk is gone from head (outdated)
        "position": None if outdated else (c.get("line") or c.get("originalLine") or 1),
        "created_at": c.get("createdAt"),
        "updated_at": c.get("updatedAt"),
        "html_url": c.get("url"),
        "diff_hunk": c.get("diffHunk"),
        "commit_id": (c.get("commit") or {}).get("oid"),
        "in_reply_to_id": (c.get("replyTo") or {}).get("databaseId"),
        "pull_request_review_id": (c.get("pullRequestReview") or {}).get("databaseId"),
        "reactions": rest_reactions(c.get("reactionGroups")),
    }


def _commit(n):
    c = n.get("commit") or {}
    a, cm = c.get("author") or {}, c.get("committer") or {}
    user = (a.get("user") or {}).get("login")
    return {
        "sha": c.get("oid"),
        "commit": {
            "message": c.get("message") or "",
            "author": {"name": a.get("name"), "email": a.get("email"), "date": _utc(a.get("date"))},
            "committer": {
                "name": cm.get("name"),
                "email": cm.get("email"),
                "date": _utc(cm.get("date")),
            },
        },
        "author": {"login": user} if user else None,
    }


def pr_lists(
    repo,
    n,
    timeout = 120,
    env = None,
):
    """{REST path: REST-shaped list} for PR `n`'s reviews, issue comments, review comments,
    commits and each issue comment's reactions, from ONE GraphQL query. A list with more pages
    than fetched is omitted (read it over REST). Raises GraphQLError."""
    owner, name = repo.split("/", 1)
    pr = (
        query(PR_LISTS_Q, timeout = timeout, env = env, owner = owner, name = name, n = int(n)).get(
            "repository"
        )
        or {}
    ).get("pullRequest")
    if pr is None:
        raise GraphQLError(f"{repo}#{n}: no such pull request")
    base, out = f"repos/{repo}", {}

    def full(conn):
        return not (conn.get("pageInfo") or {}).get("hasNextPage")

    rv = pr.get("reviews") or {}
    if full(rv):
        out[f"{base}/pulls/{n}/reviews"] = [_review(r) for r in rv.get("nodes") or []]
    ic = pr.get("comments") or {}
    if full(ic):
        nodes = ic.get("nodes") or []
        out[f"{base}/issues/{n}/comments"] = [_issue_comment(c) for c in nodes]
        for c in nodes:
            rx = rest_reaction_list(c.get("reactionGroups"))
            if rx is not None and c.get("databaseId"):
                out[f"{base}/issues/comments/{c['databaseId']}/reactions"] = rx
    th = pr.get("reviewThreads") or {}
    if full(th) and all(full(t.get("comments") or {}) for t in th.get("nodes") or []):
        rc = [
            _review_comment(c)
            for t in th.get("nodes") or []
            for c in (t.get("comments") or {}).get("nodes") or []
        ]
        out[f"{base}/pulls/{n}/comments"] = sorted(
            rc, key = lambda c: (c.get("created_at") or "", c.get("id") or 0)
        )
    cm = pr.get("commits") or {}
    if full(cm):
        out[f"{base}/pulls/{n}/commits"] = [_commit(x) for x in cm.get("nodes") or []]
    return out


def pr_diffstat(
    repo,
    n,
    timeout = 60,
    env = None,
):
    """{"additions", "deletions", "changed_files", "head_sha"} of PR `n` (GraphQL). Raises GraphQLError."""
    owner, name = repo.split("/", 1)
    q = (
        "query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name)"
        "{pullRequest(number:$n){additions deletions changedFiles headRefOid}}}"
    )
    pr = (
        query(q, timeout = timeout, env = env, owner = owner, name = name, n = int(n)).get("repository") or {}
    ).get("pullRequest")
    if pr is None:
        raise GraphQLError(f"{repo}#{n}: no such pull request")
    return {
        "additions": pr.get("additions") or 0,
        "deletions": pr.get("deletions") or 0,
        "changed_files": pr.get("changedFiles") or 0,
        "head_sha": pr.get("headRefOid"),
    }


def file_status(change_type):
    return _FILE_STATUS.get(change_type, (change_type or "").lower())


# ----------------------------------------------------------------- `gh api <REST path>` through GraphQL
# run_api(argv, fallback) answers a read-only `gh api repos/...` argv from GraphQL, in the REST
# response shape, applying `--jq` with the jq binary as gh would; anything it cannot serve (another
# path, a flag it does not know, a GraphQL error, a missing object) runs `fallback()` instead, which
# is the caller's original REST call, so a 404 or an error still looks exactly as it did.
import base64 as _b64  # noqa: E402
import re as _re  # noqa: E402
from urllib.parse import parse_qs as _parse_qs, unquote as _unquote  # noqa: E402

_PR_FIELDS = (
    "number title body state isDraft merged mergeable url createdAt updatedAt mergedAt closedAt "
    f"{_ACTOR} headRefOid headRefName headRepository {{ nameWithOwner }} baseRefOid baseRefName "
    "baseRepository { nameWithOwner } additions deletions changedFiles labels(first: 50) { nodes { name } }"
)
_ISSUE_FIELDS = (
    "number title body state url createdAt updatedAt closedAt "
    + _ACTOR
    + " labels(first: 50) { nodes { name } } assignees(first: 20) { nodes { login } } comments { totalCount }"
)
_SIG = "name email date"
# the list form: no body or diffstat (a page of 100 stays a few points); labels capped at 20
_PR_LIST_FIELDS = (
    "number title body state isDraft merged url createdAt updatedAt mergedAt closedAt "
    f"{_ACTOR} headRefOid headRefName headRepository {{ nameWithOwner }} baseRefOid baseRefName "
    "baseRepository { nameWithOwner } labels(first: 20) { nodes { name } }"
)


def _mergeable(v):
    return {"MERGEABLE": True, "CONFLICTING": False}.get(v)


def _rest_pull(p):
    head_repo, base_repo = (p.get("headRepository") or {}), (p.get("baseRepository") or {})
    return {
        "number": p.get("number"),
        "title": p.get("title"),
        "body": p.get("body"),
        "state": (p.get("state") or "").lower() if p.get("state") != "MERGED" else "closed",
        "draft": p.get("isDraft"),
        "merged": p.get("merged"),
        "mergeable": _mergeable(p.get("mergeable")),
        "html_url": p.get("url"),
        "user": rest_user(p.get("author")),
        "created_at": p.get("createdAt"),
        "updated_at": p.get("updatedAt"),
        "merged_at": p.get("mergedAt"),
        "closed_at": p.get("closedAt"),
        "head": {
            "sha": p.get("headRefOid"),
            "ref": p.get("headRefName"),
            "repo": {"full_name": head_repo.get("nameWithOwner")} if head_repo else None,
        },
        "base": {
            "sha": p.get("baseRefOid"),
            "ref": p.get("baseRefName"),
            "repo": {"full_name": base_repo.get("nameWithOwner")} if base_repo else None,
        },
        "additions": p.get("additions"),
        "deletions": p.get("deletions"),
        "changed_files": p.get("changedFiles"),
        "labels": [{"name": x.get("name")} for x in (p.get("labels") or {}).get("nodes") or []],
    }


def _rest_issue(i, is_pr):
    out = {
        "number": i.get("number"),
        "title": i.get("title"),
        "body": i.get("body"),
        "state": "closed" if i.get("state") in ("CLOSED", "MERGED") else "open",
        "html_url": i.get("url"),
        "user": rest_user(i.get("author")),
        "created_at": i.get("createdAt"),
        "updated_at": i.get("updatedAt"),
        "closed_at": i.get("closedAt"),
        "labels": [{"name": x.get("name")} for x in (i.get("labels") or {}).get("nodes") or []],
        "assignees": [
            {"login": x.get("login")} for x in (i.get("assignees") or {}).get("nodes") or []
        ],
        "comments": ((i.get("comments") or {}).get("totalCount")) or 0,
        "pull_request": None,
    }
    if is_pr:
        out["pull_request"] = {"html_url": i.get("url")}
    return out


def _utc(ts):
    """GraphQL git dates carry the committer's offset (2026-10-07T12:38:35-07:00); REST gives UTC Z."""
    from datetime import datetime, timezone
    try:
        return (
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            .astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    except (AttributeError, ValueError):
        return ts


def _sig(x):
    x = dict(x or {})
    if "date" in x:
        x["date"] = _utc(x["date"])
    return x


def _rest_commit(c):
    return {
        "sha": c.get("oid"),
        "commit": {
            "message": c.get("message") or "",
            "author": _sig(c.get("author")),
            "committer": _sig(c.get("committer")),
        },
    }


def _paged(q, conn_path, timeout, env, **v):
    """Every node of one connection (`conn_path` = keys from data to it), 100 per query."""
    nodes, after = [], None
    while True:
        d = query(q, timeout = timeout, env = env, **v, **({"after": after} if after else {}))
        conn = d
        for k in conn_path:
            conn = (conn or {}).get(k)
        if conn is None:
            raise GraphQLError("object not found")
        nodes += conn.get("nodes") or []
        pi = conn.get("pageInfo") or {}
        if not pi.get("hasNextPage"):
            return nodes
        after = pi.get("endCursor")


def rest_get(
    path,
    timeout = 60,
    env = None,
):
    """The REST JSON for a read-only `repos/...` path, from GraphQL; None when not served here.
    Raises GraphQLError when GraphQL fails or the object does not exist."""
    path, _, qs = path.lstrip("/").partition("?")
    params = {k: v[-1] for k, v in _parse_qs(qs).items()}
    if path == "user" and not params:
        v = (
            query(
                "query { rateLimit { cost remaining resetAt } viewer { id databaseId login name } }",
                timeout = timeout,
                env = env,
            ).get("viewer")
            or {}
        )
        if not v.get("login"):
            raise GraphQLError("no viewer")
        return {
            "login": v["login"],
            "id": v.get("databaseId"),
            "node_id": v.get("id"),
            "name": v.get("name"),
            "type": "User",
        }
    m = _re.match(r"^repos/([^/]+)/([^/]+)(?:/(.*))?$", path)
    if not m:
        return None
    o, r, rest = m[1], m[2], m[3] or ""
    V = {"owner": o, "name": r}
    REPO = "query($owner: String!, $name: String!%s) { rateLimit { cost remaining resetAt } repository(owner: $owner, name: $name) { %s } }"

    def one(extra_vars, body, **v):
        d = query(REPO % (extra_vars, body), timeout = timeout, env = env, **V, **v)
        repo = d.get("repository")
        if repo is None:
            raise GraphQLError(f"{o}/{r}: repository not found")
        return repo

    if rest == "" and not params:
        x = one(
            "",
            "nameWithOwner url isPrivate isArchived isFork visibility autoMergeAllowed defaultBranchRef { name }",
        )
        return {
            "full_name": x["nameWithOwner"],
            "html_url": x["url"],
            "private": x["isPrivate"],
            "archived": x["isArchived"],
            "fork": x["isFork"],
            "visibility": (x.get("visibility") or "").lower(),
            "allow_auto_merge": x["autoMergeAllowed"],
            "default_branch": (x.get("defaultBranchRef") or {}).get("name"),
        }
    if (mm := _re.fullmatch(r"pulls/(\d+)", rest)) and not params:
        p = one(", $n: Int!", f"pullRequest(number: $n) {{ {_PR_FIELDS} }}", n = int(mm[1])).get(
            "pullRequest"
        )
        if p is None:
            raise GraphQLError("pull request not found")
        return _rest_pull(p)
    if (
        rest == "pulls"
        and set(params) <= {"state", "per_page"}
        and params.get("state", "open") in ("open", "closed", "all")
    ):
        states = {"open": "[OPEN]", "closed": "[CLOSED, MERGED]", "all": "[OPEN, CLOSED, MERGED]"}[
            params.get("state", "open")
        ]
        q = REPO % (
            ", $after: String",
            f"pullRequests(states: {states}, first: 100, after: $after, "
            "orderBy: {field: CREATED_AT, direction: DESC}) { pageInfo { hasNextPage endCursor } "
            f"nodes {{ {_PR_LIST_FIELDS} }} }}",
        )
        return [_rest_pull(p) for p in _paged(q, ("repository", "pullRequests"), timeout, env, **V)]
    if (mm := _re.fullmatch(r"issues/(\d+)", rest)) and not params:
        i = one(
            ", $n: Int!",
            f"issueOrPullRequest(number: $n) {{ __typename ... on Issue {{ {_ISSUE_FIELDS} }} "
            f"... on PullRequest {{ {_ISSUE_FIELDS} }} }}",
            n = int(mm[1]),
        ).get("issueOrPullRequest")
        if i is None:
            raise GraphQLError("issue not found")
        return _rest_issue(i, i.get("__typename") == "PullRequest")
    if (
        (mm := _re.fullmatch(r"pulls/(\d+)/files", rest))
        and set(params) <= {"per_page", "page"}
        and "page" not in params
    ):
        q = REPO % (
            ", $n: Int!, $after: String",
            "pullRequest(number: $n) { files(first: 100, after: $after) "
            "{ pageInfo { hasNextPage endCursor } nodes { path additions deletions changeType } } }",
        )
        nodes = _paged(q, ("repository", "pullRequest", "files"), timeout, env, **V, n = int(mm[1]))
        return [
            {
                "filename": f["path"],
                "status": file_status(f.get("changeType")),
                "additions": f["additions"],
                "deletions": f["deletions"],
                "changes": f["additions"] + f["deletions"],
            }
            for f in nodes
        ]
    if (mm := _re.fullmatch(r"issues/(\d+)/comments", rest)) and set(params) <= {"per_page"}:
        q = REPO % (
            ", $n: Int!, $after: String",
            "issueOrPullRequest(number: $n) { ... on Issue { comments(first: 100, after: $after) "
            "{ pageInfo { hasNextPage endCursor } nodes { id databaseId "
            + _ACTOR
            + " body createdAt updatedAt url "
            + _COUNTS
            + " } } } ... on PullRequest { comments(first: 100, after: $after) { pageInfo { hasNextPage endCursor } "
            "nodes { id databaseId "
            + _ACTOR
            + " body createdAt updatedAt url "
            + _COUNTS
            + " } } } }",
        )
        return [
            _issue_comment(c)
            for c in _paged(
                q, ("repository", "issueOrPullRequest", "comments"), timeout, env, **V, n = int(mm[1])
            )
        ]
    if (mm := _re.fullmatch(r"pulls/(\d+)/(reviews|comments|commits)", rest)) and set(params) <= {
        "per_page"
    }:
        got = pr_lists(f"{o}/{r}", int(mm[1]), timeout = timeout, env = env).get(
            f"repos/{o}/{r}/{rest}"
        )
        if got is None:
            raise GraphQLError("more items than one query fetches")
        return got
    if (mm := _re.fullmatch(r"commits/([^/]+)", rest)) and not params:
        c = one(
            ", $ref: String!",
            f"object(expression: $ref) {{ ... on Commit {{ oid message author {{ {_SIG} }} "
            f"committer {{ {_SIG} }} }} }}",
            ref = _unquote(mm[1]),
        ).get("object")
        if not c or not c.get("oid"):
            raise GraphQLError("commit not found")
        return _rest_commit(c)
    if rest == "commits" and set(params) <= {"sha", "per_page"} and params.get("sha"):
        k = min(int(params.get("per_page") or 30), 100)
        c = one(
            ", $ref: String!, $k: Int!",
            f"object(expression: $ref) {{ ... on Commit {{ history(first: $k) {{ nodes "
            f"{{ oid message author {{ {_SIG} }} committer {{ {_SIG} }} }} }} }} }}",
            ref = params["sha"],
            k = k,
        ).get("object")
        if not c or "history" not in c:
            raise GraphQLError("ref not found")
        return [_rest_commit(x) for x in c["history"]["nodes"]]
    if (mm := _re.fullmatch(r"git/ref/heads/(.+)", rest)) and not params:
        ref = one(
            ", $qn: String!",
            "ref(qualifiedName: $qn) { target { oid } }",
            qn = f"refs/heads/{_unquote(mm[1])}",
        ).get("ref")
        if not ref:
            raise GraphQLError("ref not found")
        return {
            "ref": f"refs/heads/{_unquote(mm[1])}",
            "object": {"sha": ref["target"]["oid"], "type": "commit"},
        }
    if (mm := _re.fullmatch(r"contents/(.+)", rest)) and set(params) <= {"ref"}:
        fp = _unquote(mm[1])
        b = one(
            ", $e: String!",
            "object(expression: $e) { __typename ... on Blob { oid byteSize isBinary isTruncated text } }",
            e = f"{params.get('ref', 'HEAD')}:{fp}",
        ).get("object")
        if (
            not b
            or b.get("__typename") != "Blob"
            or b.get("isBinary")
            or b.get("isTruncated")
            or b.get("text") is None
        ):
            raise GraphQLError("not a text blob GraphQL can return whole")
        return {
            "type": "file",
            "name": fp.rsplit("/", 1)[-1],
            "path": fp,
            "sha": b["oid"],
            "size": b["byteSize"],
            "encoding": "base64",
            "content": _b64.b64encode(b["text"].encode()).decode(),
        }
    return None


def _parse_api_argv(argv):
    """(path, jq) for `gh api <path> [--paginate] [--jq X | -q X] [-H Accept...]`, else None."""
    if argv[:2] != ["gh", "api"]:
        return None
    path, jq, i = None, None, 2
    while i < len(argv):
        a = argv[i]
        if a in ("--jq", "-q") and i + 1 < len(argv):
            jq, i = argv[i + 1], i + 2
            continue
        if a == "--paginate":
            i += 1
            continue
        if (
            a == "-H"
            and i + 1 < len(argv)
            and argv[i + 1].lower().startswith("accept: application/vnd.github+json")
        ):
            i += 2
            continue
        if a.startswith("-") or path is not None:
            return None  # -X / -f / -i / anything else: not a plain read
        path, i = a, i + 1
    return (
        (path, jq)
        if path and (path.lstrip("/").startswith("repos/") or path.lstrip("/") == "user")
        else None
    )


def _gh_print(v):
    """One --jq result the way gh prints it (not `jq -r`): strings raw, a top-level null as an empty
    line, anything else compact JSON with sorted keys and Go's HTML escaping."""
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


def _projected(argv, timeout, env):
    """gh's stdout for `argv` from gh_translate, which proves the --jq reads only fields it
    reproduces and fetches just those (a PR's mergeable / body are most of a full query's time);
    None when it does not cover the call."""
    try:
        import gh_translate
    except ImportError:
        return None
    plan = gh_translate.translate(list(argv), env or os.environ)
    if plan is None:
        return None
    run_env = dict(env or os.environ, GH_GRAPHQL_TRANSLATE = "never")

    def runner(args):
        r = subprocess.run(
            ["gh", *args], capture_output = True, text = True, timeout = timeout, env = run_env
        )
        return r.returncode, r.stdout, r.stderr

    got = gh_translate.run(plan, runner)
    return got[0] if got else None


def run_api(
    argv,
    fallback,
    timeout = 60,
    env = None,
):
    """CompletedProcess for a read-only `gh api repos/...` argv answered from GraphQL (REST JSON,
    `--jq` applied like gh does), or `fallback()` (the caller's own REST call) when it cannot be."""
    parsed = _parse_api_argv(list(argv)) if enabled() else None
    if parsed:
        path, jq = parsed
        got = _projected(argv, timeout, env) if jq is not None else None
        if got is not None:
            return subprocess.CompletedProcess(argv, 0, got, "")
        try:
            data = rest_get(path, timeout = timeout, env = env)
        except GraphQLError as e:
            if os.environ.get("GH_GRAPHQL_DEBUG"):
                print(f"gh_graphql: {path}: {e}; REST", file = sys.stderr)
            data = None
        if data is not None:
            text = json.dumps(data)
            if jq is None:
                return subprocess.CompletedProcess(argv, 0, text + "\n", "")
            try:
                j = subprocess.run(
                    ["jq", "-c", jq], input = text, capture_output = True, text = True, timeout = 30
                )
            except (OSError, subprocess.TimeoutExpired):
                j = None
            if j is not None and j.returncode == 0:
                try:
                    out = "".join(
                        _gh_print(json.loads(ln)) + "\n"
                        for ln in j.stdout.splitlines()
                        if ln.strip()
                    )
                except ValueError:
                    out = None
                if out is not None:
                    return subprocess.CompletedProcess(argv, 0, out, "")
    return fallback()


def try_api(
    argv,
    timeout = 60,
    env = None,
):
    """run_api without a fallback: the CompletedProcess when GraphQL served `argv`, else None
    (the caller then makes its REST call exactly as before)."""
    return run_api(argv, lambda: None, timeout = timeout, env = env)


if __name__ == "__main__":
    # `python3 gh_graphql.py api <gh api args>`: print what GraphQL served, exit 3 when it could not
    # (the shell caller then makes its REST call as before).
    if sys.argv[1:2] != ["api"]:
        print(__doc__.split("\n\n")[0], file = sys.stderr)
        raise SystemExit(2)
    got = try_api(["gh", *sys.argv[1:]])
    if got is None:
        raise SystemExit(3)
    sys.stdout.write(got.stdout)
