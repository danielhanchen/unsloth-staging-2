import runpy, pathlib
def test_run_under_pytest():
    runpy.run_path(str(pathlib.Path(__file__).with_name("spawn_exp.py")))
