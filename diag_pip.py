"""pip --target inside MXC: full traceback, with the default temp and with an existing temp dir."""
import os, subprocess, sys
sys.path.insert(0, "studio/backend")
from core.inference import tools
for arm in ("default-temp", "workdir-temp"):
    session = f"__LOCALID_pipdiag_{arm.replace('-', '_')}"
    workdir = tools._get_workdir(session)
    wheels = os.path.join(workdir, "wheels")
    os.makedirs(wheels, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "-d", wheels, "six==1.16.0"], check=False)
    fix = "os.environ['TEMP'] = os.environ['TMP'] = os.environ['TMPDIR']; tempfile.tempdir = None\n" if arm == "workdir-temp" else ""
    code = (
        "import os, tempfile, traceback, logging\n" + fix +
        "print('TEMPDIR', tempfile.gettempdir())\n"
        "from pip._internal.cli.main import main\n"
        "w = os.path.join(os.getcwd(), 'wheels')\n"
        "rc = main(['install', '-v', '--no-deps', '--no-index', '--find-links', w, '--disable-pip-version-check', '--log', os.path.join(os.getcwd(), 'pip.log'), 'six==1.16.0'])\n"
        "print('PIP_EXIT', rc)\n"
    )
    print("=== ARM", arm)
    print(tools._python_exec(code, None, 300, session, tool_execution_mode="required")[-3000:])
    log = os.path.join(workdir, "pip.log")
    if os.path.exists(log):
        text = open(log, encoding="utf-8", errors="replace").read()
        print("--- pip.log tail ---")
        print(text[-4000:])
    pk = os.path.join(workdir, ".unsloth-packages")
    print("HOST packages:", os.listdir(pk) if os.path.isdir(pk) else "none")
