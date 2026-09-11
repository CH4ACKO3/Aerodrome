"""Engine-neutral message bridge; no WebSocket/UE runtime dependency."""
from dataclasses import asdict
from typing import Protocol
import json
from .backend import Capabilities, RenderReceipt


class RenderTransport(Protocol):
    """Implement I/O + timeouts in an adapter. Messages are JSON strings.

    exchange is a bounded-time startup handshake; send/receive are nonblocking
    and must use bounded queues. receive returns one message or None. close
    cancels pending operations. One connection per backend session.
    """
    def exchange(self,message: str) -> str: ...
    def send(self,message: str) -> None: ...
    def receive(self) -> str | None: ...
    def close(self) -> None: ...


class MessageBackend:
    """Adapts the renderer lifecycle to a transport implemented by the user."""
    def __init__(self,transport):
        self.transport = transport
        self._open = False

    def open(self,scene,config):
        if self._open:
            raise RuntimeError("backend already open")
        message = dict(protocol_version=1,kind="open",scene=scene.to_dict(),config=asdict(config))
        reply = json.loads(self.transport.exchange(json.dumps(message,allow_nan=False)))
        if reply.get("protocol_version")!=1 or reply.get("kind")!="ready":
            raise ValueError("renderer protocol handshake failed")
        caps = reply["capabilities"]
        modes = tuple(caps.get("modes",()))
        if not modes or any(x not in ("live","offline") for x in modes):
            raise ValueError("invalid renderer modes")
        if any(type(caps.get(k,False)) is not bool for k in ("camera","articulations")):
            raise ValueError("invalid capability flags")
        self._open = True
        return Capabilities(modes,caps.get("articulations",False),caps.get("camera",False))

    def submit(self,request):
        if not self._open:
            raise RuntimeError("backend not open")
        self.transport.send(json.dumps(request.to_dict(),allow_nan=False,separators=(",",":")))

    def poll(self):
        message = self.transport.receive()
        if message is None:
            return ()
        reply = json.loads(message)
        if reply.pop("protocol_version",None)!=1 or reply.pop("kind",None)!="receipt":
            raise ValueError("invalid renderer receipt envelope")
        return (RenderReceipt(**reply),)

    def close(self):
        self._open = False
        self.transport.close()
