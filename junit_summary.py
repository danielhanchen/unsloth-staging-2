import os
import sys
import xml.etree.ElementTree as E

root = E.parse(os.path.join(os.environ["RUNNER_TEMP"], sys.argv[1] if len(sys.argv) > 1 else "junit.xml")).getroot()
for case in root.iter("testcase"):
    if case.find("failure") is not None or case.find("error") is not None:
        print("FAILED", case.get("classname"), case.get("name"))
suite = next(root.iter("testsuite"))
print("JUNIT", {k: suite.get(k) for k in ("tests", "failures", "errors", "skipped")})
