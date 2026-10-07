import subprocess
import unittest
from unittest import mock

from atomic_memory import paths


HEADER = "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
NAT_ROUTE = "eth0 00000000 017011AC 0003 0 0 0 00000000 0 0 0\n"
LOCAL_URL = "http://127.0.0.1:11434"
HOST_URL = "http://172.17.112.1:11434"


class OllamaAddressTest(unittest.TestCase):
    def resolve(self, system="Linux", kernel="6.6-microsoft-standard-WSL2",
                routes=HEADER + NAT_ROUTE, mode="nat", mode_error=None):
        def read_proc(path, **kwargs):
            if path.as_posix() == "/proc/sys/kernel/osrelease":
                value = kernel
            else:
                value = routes
            if isinstance(value, Exception):
                raise value
            return value

        command = mock.Mock(returncode=0, stdout=mode)
        with mock.patch.dict(paths.os.environ, {}, clear=True), \
                mock.patch.object(paths.platform, "system", return_value=system), \
                mock.patch.object(paths.Path, "read_text", autospec=True,
                                  side_effect=read_proc), \
                mock.patch.object(paths.subprocess, "run", return_value=command,
                                  side_effect=mode_error):
            return paths.resolve_ollama_url()

    def test_explicit_address_takes_priority_without_platform_detection(self):
        with mock.patch.dict(paths.os.environ,
                             {"ATOMICUPDATE_OLLAMA_URL": "http://server:11434"}), \
                mock.patch.object(paths.platform, "system") as platform_check:
            self.assertEqual(paths.resolve_ollama_url(), "http://server:11434")
            platform_check.assert_not_called()

    def test_windows_uses_localhost(self):
        self.assertEqual(self.resolve(system="Windows"), LOCAL_URL)

    def test_native_linux_uses_localhost(self):
        self.assertEqual(self.resolve(kernel="6.8.0-generic"), LOCAL_URL)

    def test_wsl_nat_decodes_gateway_in_little_endian_order(self):
        self.assertEqual(self.resolve(), HOST_URL)

    def test_wsl_mirrored_uses_localhost(self):
        self.assertEqual(self.resolve(mode="mirrored\n"), LOCAL_URL)

    def test_older_wsl_without_wslinfo_uses_gateway(self):
        self.assertEqual(self.resolve(mode_error=FileNotFoundError()), HOST_URL)

    def test_wslinfo_timeout_still_uses_gateway(self):
        self.assertEqual(self.resolve(mode_error=subprocess.TimeoutExpired(
            "wslinfo", 2)), HOST_URL)

    def test_lowest_metric_active_default_route_is_selected(self):
        routes = HEADER + NAT_ROUTE.replace("0 0 0 00000000", "0 0 10 00000000")
        routes += "eth1 00000000 0100A8C0 0003 0 0 5 00000000 0 0 0\n"
        routes += "eth2 00000000 0100000A 0002 0 0 0 00000000 0 0 0\n"
        self.assertEqual(self.resolve(routes=routes), "http://192.168.0.1:11434")

    def test_unavailable_kernel_or_route_falls_back_to_localhost(self):
        self.assertEqual(self.resolve(kernel=OSError()), LOCAL_URL)
        self.assertEqual(self.resolve(routes=OSError()), LOCAL_URL)

    def test_malformed_and_nondefault_routes_are_ignored(self):
        routes = HEADER + "eth0 00000000 invalid 0003 0 0 0 00000000\n"
        routes += "eth0 00000000 00000000 0003 0 0 0 00000000\n"
        routes += "eth0 00000001 017011AC 0003 0 0 0 00000000\n"
        self.assertEqual(self.resolve(routes=routes), LOCAL_URL)


if __name__ == "__main__":
    unittest.main()
