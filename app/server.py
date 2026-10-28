"""Run Uvicorn with a listening socket that survives idle recycling."""

import os
import socket

import uvicorn


def main() -> None:
    fd = os.environ.get("NEMOTRON_LISTEN_FD")
    if fd is None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 8000))
        listener.listen(2048)
        listener.set_inheritable(True)
        os.environ["NEMOTRON_LISTEN_FD"] = str(listener.fileno())
    else:
        listener = socket.socket(fileno=int(fd))

    config = uvicorn.Config("app.main:app", host="0.0.0.0", port=8000)
    uvicorn.Server(config).run(sockets=[listener])


if __name__ == "__main__":
    main()
