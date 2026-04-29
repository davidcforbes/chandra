"""Unit tests for chandra.scripts.vllm — GPU sizing & sudo detection."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from chandra.scripts import vllm as vllm_script


# ---------- get_gpu_settings ----------


class TestGetGpuSettings:
    def test_h100_baseline(self):
        tokens, seqs = vllm_script.get_gpu_settings("h100")
        assert tokens == vllm_script.BASELINE_MAX_BATCHED_TOKENS
        assert seqs == vllm_script.BASELINE_MAX_NUM_SEQS

    def test_smaller_gpu_gets_smaller_settings(self):
        h100_tokens, h100_seqs = vllm_script.get_gpu_settings("h100")
        t4_tokens, t4_seqs = vllm_script.get_gpu_settings("t4")
        assert t4_tokens < h100_tokens
        assert t4_seqs < h100_seqs

    def test_unknown_gpu_raises(self):
        with pytest.raises(SystemExit, match="Unknown GPU"):
            vllm_script.get_gpu_settings("imaginary-gpu-9000")


# ---------- docker_invocation (chandra-730) ----------


class TestDockerInvocation:
    def test_no_sudo_when_docker_works(self):
        # docker is on PATH and `docker info` succeeds → no sudo.
        with patch.object(vllm_script.shutil, "which", return_value="/bin/docker"):
            with patch.object(vllm_script.subprocess, "run") as run:
                run.return_value.returncode = 0
                cmd = vllm_script.docker_invocation()
        assert cmd == ["docker"]

    def test_falls_back_to_sudo_when_docker_info_fails(self):
        # First `which` (docker) → present; second `which` (sudo) → present.
        which_calls = iter(["/bin/docker", "/bin/sudo"])
        with patch.object(
            vllm_script.shutil, "which", side_effect=lambda b: next(which_calls)
        ):
            with patch.object(vllm_script.subprocess, "run") as run:
                run.return_value.returncode = 1  # docker info failed
                cmd = vllm_script.docker_invocation()
        assert cmd == ["sudo", "docker"]

    def test_no_docker_raises(self):
        with patch.object(vllm_script.shutil, "which", return_value=None):
            with pytest.raises(SystemExit, match="docker binary"):
                vllm_script.docker_invocation()

    def test_no_sudo_no_elevation(self):
        # docker info fails AND sudo is missing → return docker_bin alone and
        # let the actual run surface the platform error.
        which_calls = iter(["/bin/docker", None])
        with patch.object(
            vllm_script.shutil, "which", side_effect=lambda b: next(which_calls)
        ):
            with patch.object(vllm_script.subprocess, "run") as run:
                run.return_value.returncode = 1
                cmd = vllm_script.docker_invocation()
        assert cmd == ["docker"]

    def test_handles_subprocess_exception(self):
        # If subprocess.run raises (FileNotFoundError on misbehaving systems),
        # we should still return *some* invocation — fall back to sudo if
        # available.
        which_calls = iter(["/bin/docker", "/bin/sudo"])
        with patch.object(
            vllm_script.shutil, "which", side_effect=lambda b: next(which_calls)
        ):
            with patch.object(
                vllm_script.subprocess, "run", side_effect=FileNotFoundError("no")
            ):
                cmd = vllm_script.docker_invocation()
        assert cmd == ["sudo", "docker"]


# ---------- build_command ----------


class TestBuildCommand:
    def _args(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            gpu="h100",
            mtp=False,
            docker_bin="docker",
            gpu_runtime="nvidia",
            image="my-image:latest",
            port=8000,
            print_only=True,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_includes_image_and_port_mapping(self):
        with patch.object(vllm_script, "docker_invocation", return_value=["docker"]):
            cmd = vllm_script.build_command(self._args(), 8192, 64)
        assert "my-image:latest" in cmd
        assert "8000:8000" in " ".join(cmd)

    def test_omits_runtime_when_blank(self):
        with patch.object(vllm_script, "docker_invocation", return_value=["docker"]):
            cmd = vllm_script.build_command(self._args(gpu_runtime=""), 8192, 64)
        assert "--runtime" not in cmd

    def test_appends_mtp_config(self):
        with patch.object(vllm_script, "docker_invocation", return_value=["docker"]):
            cmd = vllm_script.build_command(self._args(mtp=True), 8192, 64)
        assert "--speculative-config" in cmd


# ---------- parse_args ----------


class TestParseArgs:
    def test_defaults(self):
        ns = vllm_script.parse_args([])
        assert ns.gpu == "h100"
        assert ns.docker_bin == "docker"
        assert ns.image.startswith("vllm/")
        assert ns.port == 8000
        assert ns.mtp is False

    def test_print_only(self):
        ns = vllm_script.parse_args(["--print-only"])
        assert ns.print_only is True


# ---------- main() smoke test ----------


class TestMain:
    def test_print_only_does_not_run_subprocess(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["chandra_vllm", "--print-only"])
        with patch.object(vllm_script, "docker_invocation", return_value=["docker"]):
            with patch.object(vllm_script.subprocess, "run") as fake_run:
                vllm_script.main()
        fake_run.assert_not_called()

    def test_main_invokes_subprocess(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["chandra_vllm", "--gpu", "t4"])
        with patch.object(vllm_script, "docker_invocation", return_value=["docker"]):
            with patch.object(vllm_script.subprocess, "run") as fake_run:
                vllm_script.main()
        fake_run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
