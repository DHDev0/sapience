"""The tool/plugin registry: register, run, toggle, persist."""
import os, shutil
from brain.tools import ToolRegistry


def test_add_run_persist():
    d = "/tmp/_tools_test"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    p = os.path.join(d, "tools.json")
    t = ToolRegistry(p)
    r = t.add({"name": "echo", "cmd": "echo hello {input}", "kind": "text", "autonomous": True})
    assert r["ok"] and t.list()[0]["name"] == "echo"
    assert t.autonomous() == ["echo"]                            # enabled + autonomous
    out = t.run("echo", "world")
    assert out["ok"] and "hello world" in out["output"]
    # persistence: reload
    t2 = ToolRegistry(p)
    assert [x["name"] for x in t2.list()] == ["echo"]
    shutil.rmtree(d, ignore_errors=True)


def test_toggle_and_remove():
    d = "/tmp/_tools_test2"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    t = ToolRegistry(os.path.join(d, "tools.json"))
    t.add({"name": "a", "cmd": "echo {input}"})
    t.toggle("a", enabled=False)
    assert t.run("a", "x")["ok"] is False                       # disabled tool won't run
    t.remove("a")
    assert t.list() == []
    shutil.rmtree(d, ignore_errors=True)


def test_shell_and_missing_command():
    d = "/tmp/_tools_test3"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    t = ToolRegistry(os.path.join(d, "tools.json"))
    t.add({"name": "sh", "cmd": "printf '%s' {input}", "shell": True})
    assert t.run("sh", "abc")["output"] == "abc"
    t.add({"name": "nope", "cmd": "this_command_does_not_exist_xyz {input}"})
    assert t.run("nope", "x")["ok"] is False                    # graceful failure, no crash
    shutil.rmtree(d, ignore_errors=True)
