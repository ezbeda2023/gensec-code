"""Regression checks for browsers disconnecting during startup requests."""

from http.client import HTTPConnection
from io import BytesIO
import unittest
from unittest.mock import Mock

from startup_screen import LoadingHandler, LoadingScreen, PAGE


class StartupScreenTests(unittest.TestCase):
    def request(self, send_error=None, read_error=None):
        connection = Mock()
        connection.makefile.return_value = BytesIO(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
        if read_error:
            connection.makefile.return_value = Mock()
            connection.makefile.return_value.readline.side_effect = read_error
        connection.sendall.side_effect = send_error
        return LoadingHandler(connection, ('127.0.0.1', 12345), Mock())

    def test_disconnects_during_request_headers_and_body_are_quiet(self):
        for error_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            for phase in ('read', 'headers', 'body'):
                with self.subTest(error=error_type.__name__, phase=phase):
                    error = error_type('Browser disconnected')
                    if phase == 'read':
                        handler = self.request(read_error=error)
                    else:
                        handler = self.request(send_error=error if phase == 'headers'
                                               else [None, error])
                    self.assertTrue(handler.close_connection)

    def test_unexpected_errors_remain_visible(self):
        with self.assertRaisesRegex(OSError, 'Unexpected failure'):
            self.request(send_error=OSError('Unexpected failure'))

    def test_loading_page_still_serves_get_and_head(self):
        with LoadingScreen('127.0.0.1', 0) as screen:
            for method in ('GET', 'HEAD'):
                with self.subTest(method=method):
                    client = HTTPConnection(*screen.server.server_address, timeout=5)
                    try:
                        client.request(method, '/')
                        response = client.getresponse()
                        self.assertEqual(response.status, 202)
                        self.assertEqual(response.getheader('Cache-Control'), 'no-store')
                        self.assertEqual(response.read(), PAGE if method == 'GET' else b'')
                    finally:
                        client.close()


if __name__ == '__main__':
    unittest.main()
