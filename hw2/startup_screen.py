"""Serve a temporary page while the heavier Chainlit dependencies load."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import ssl
from threading import Thread


PAGE = b'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Please wait | Source Notebook</title>
  <style>
    :root { color-scheme: light dark; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center;
           background: #101827; color: #edf2fa; font: 18px system-ui, sans-serif; }
    main { text-align: center; padding: 2rem; }
    h1 { font-size: 2rem; font-weight: 600; margin: 1.5rem 0 .5rem; }
    p { color: #b9c5d8; line-height: 1.6; }
    .spinner { width: 42px; height: 42px; margin: auto; border: 4px solid #334155;
               border-top-color: #93c5fd; border-radius: 50%;
               animation: spin 1s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
  </style>
</head>
<body>
  <main role="status" aria-live="polite">
    <div class="spinner" aria-hidden="true"></div>
    <h1>Please wait</h1>
    <p id="status">Source Notebook is starting.<br>This page will open the chat automatically.</p>
    <noscript><p>Refresh this page once the terminal says the app is available.</p></noscript>
  </main>
  <script>
    const started = Date.now();
    async function checkReady() {
      try {
        const response = await fetch(window.location.href, {
          cache: 'no-store', signal: AbortSignal.timeout(3000)
        });
        await response.text(); // Finish reading the response before the next poll.
        if (response.ok && response.status !== 202) {
          window.location.reload();
          return;
        }
      } catch (_) { /* The port briefly closes during server handoff. */ }
      if (Date.now() - started > 120000) {
        document.getElementById('status').textContent =
          'Still waiting. Check the terminal for startup progress or errors. You can leave this page open.';
      }
      setTimeout(checkReady, 1000);
    }
    setTimeout(checkReady, 1000);
  </script>
</body>
</html>'''


class LoadingHandler(BaseHTTPRequestHandler):
    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Refreshes, closed tabs, and polling timeouts can disconnect at
            # any point: reading the request, sending headers, or writing HTML.
            self.close_connection = True

    def do_HEAD(self):
        self.send_response(202)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(PAGE)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def do_GET(self):
        self.do_HEAD()
        self.wfile.write(PAGE)

    def log_message(self, format, *args):
        pass


class LoadingScreen:
    def __init__(self, host, port, certfile=None, keyfile=None):
        class Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6 if ':' in host else socket.AF_INET
            # Do not take over an address already occupied by another app.
            allow_reuse_address = False
            daemon_threads = True

        self.server = Server((host, int(port)), LoadingHandler)
        try:
            if certfile:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(certfile, keyfile)
                self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        except Exception:
            self.server.server_close()
            raise
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
