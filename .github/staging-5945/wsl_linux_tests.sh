#!/usr/bin/env bash
# Run the Linux install-path shell tests from inside a real WSL distro.
#
# Invoked as: wsl.exe -d <distro> -- bash <repo>/.github/staging-5945/wsl_linux_tests.sh <repo>
# $1 is the repo path translated into the WSL filesystem by wslpath.
set -u

cd "${1:-.}" || exit 1

echo "--- /proc/version ---"
cat /proc/version
if grep -qi microsoft /proc/version; then
    echo "RESULT PASS install.sh would set OS=wsl on this kernel"
else
    echo "::error::/proc/version does not look like WSL; install.sh would not take the WSL branch"
    exit 1
fi

echo "--- /dev/dxg (WSL GPU paravirt device) ---"
ls -l /dev/dxg 2>/dev/null || echo "/dev/dxg: <absent> (expected: hosted runners have no GPU)"

# Same single documented skip as the other staging workflows:
# test_install_host_defaults.sh already fails on upstream main at this merge base.
skip="test_install_host_defaults.sh"
found=0
failed=""
for s in tests/sh/test_*.sh; do
    case " $skip " in
        *" $(basename "$s") "*) echo "skipping $s (see workflow comment)"; continue ;;
    esac
    found=$((found + 1))
    echo "::group::$s"
    if bash "$s" > /tmp/wsl_sh_out 2>&1; then
        echo "RESULT PASS $s"
    else
        echo "RESULT FAIL $s"
        tail -30 /tmp/wsl_sh_out
        echo "::error file=$s::shell installer test failed under real WSL"
        failed="$failed $s"
    fi
    echo "::endgroup::"
done

[ "$found" -gt 0 ] || { echo "::error::no shell tests discovered under tests/sh"; exit 1; }
echo "ran $found shell installer test files under WSL"
if [ -n "$failed" ]; then
    echo "FAILED UNDER WSL:$failed"
    exit 1
fi
