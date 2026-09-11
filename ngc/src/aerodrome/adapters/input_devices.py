"""Keyboard/mouse event bridge and opt-in camera adapter. No capture on import."""
from threading import Lock, get_ident
import numpy as np
from .external import Port
from .device_io import InputChannel


class KeyboardMouse:
    """Named keys, cumulative edge/motion counters, current held states.

    Browser/UE adapters call key/button/motion/wheel/focus on their event loop.
    Cumulative counters preserve short taps even if snapshots are overwritten;
    ordering of multiple events is not preserved. Consumers difference counts
    once per io_sequence. Button indices: left=0, middle=1, right=2.
    """
    def __init__(self,keys, **channel_options):
        self.keys = tuple(keys)
        if not self.keys or len(set(self.keys))!=len(self.keys):
            raise ValueError("keys must be nonempty and unique")
        n = len(self.keys)
        specs = (("keys",(n,),"bool"),("presses",(n,),"int32"),("releases",(n,),"int32"),
                 ("buttons",(3,),"bool"),("button_presses",(3,),"int32"),("button_releases",(3,),"int32"),
                 ("position",(2,),"float32"),("motion_total",(2,),"float32"),
                 ("wheel_total",(2,),"float32"),("focused",(),"bool"))
        self.channel = InputChannel(tuple(Port(k,"pixel" if k in ("position","motion_total") else "1",shape,"input_device",dtype=dtype) for k,shape,dtype in specs),**channel_options)
        self._values = {k:np.zeros(shape,dtype) for k,shape,dtype in specs}
        self._values["focused"][...] = True
        self._lock = Lock()

    def key(self,name,down):
        if name not in self.keys:
            return
        with self._lock:
            i = self.keys.index(name)
            if down and not self._values["focused"]:
                return
            if bool(self._values["keys"][i])!=bool(down):
                counter = "presses" if down else "releases"
                if self._values[counter][i]==np.iinfo(np.int32).max:
                    raise OverflowError("key counter exhausted")
                self._values[counter][i] += 1
                self._values["keys"][i] = down
            self.channel.publish(self._values)

    def button(self,index,down):
        if type(index) is not int or not 0<=index<3:
            raise ValueError("button index must be 0,1,2")
        with self._lock:
            down = bool(down) if self._values["focused"] else False
            if bool(self._values["buttons"][index])!=down:
                counter = "button_presses" if down else "button_releases"
                if self._values[counter][index]==np.iinfo(np.int32).max:
                    raise OverflowError("button counter exhausted")
                self._values[counter][index] += 1
                self._values["buttons"][index] = down
            self.channel.publish(self._values)

    def motion(self,position,delta):
        position,delta = np.asarray(position,np.float32),np.asarray(delta,np.float32)
        if position.shape!=(2,) or delta.shape!=(2,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(delta)):
            raise ValueError("mouse position/delta must be finite 2-vectors")
        with self._lock:
            self._values["position"][:] = position
            if self._values["focused"]:
                self._values["motion_total"] += np.asarray(delta,np.float32)
            self.channel.publish(self._values)

    def wheel(self,delta):
        delta = np.asarray(delta,np.float32)
        if delta.shape!=(2,) or not np.all(np.isfinite(delta)):
            raise ValueError("wheel delta must be a finite 2-vector")
        with self._lock:
            if self._values["focused"]:
                self._values["wheel_total"] += np.asarray(delta,np.float32)
            self.channel.publish(self._values)

    def focus(self,active):
        with self._lock:
            self._values["focused"][...] = active
            if not active:
                for counter,held in (("releases","keys"),("button_releases","buttons")):
                    if np.any(self._values[counter][self._values[held]]==np.iinfo(np.int32).max):
                        raise OverflowError("release counter exhausted")
                self._values["releases"] += self._values["keys"].astype(np.int32)
                self._values["button_releases"] += self._values["buttons"].astype(np.int32)
                self._values["keys"][:] = False
                self._values["buttons"][:] = False
            self.channel.publish(self._values)

    def heartbeat(self):
        """Call each UI poll even with no events, so held keys stay connected."""
        with self._lock:
            self.channel.publish(self._values)


class PygameInput:
    """Optional local-window adapter; caller owns pygame init/display/event loop.

    key_names maps pygame key integers to KeyboardMouse names. Does not steal
    events: pass the event list already obtained by the application.
    """
    def __init__(self,hub,key_names, *, pygame_module=None):
        if pygame_module is None:
            import pygame as pygame_module
        self.pg,self.hub,self.key_names = pygame_module,hub,dict(key_names)
        self._owner = get_ident()

    def handle(self,events):
        if get_ident()!=self._owner:
            raise RuntimeError("pygame events must be handled on owner UI thread")
        p = self.pg
        for e in events:
            if e.type in (p.KEYDOWN,p.KEYUP) and e.key in self.key_names:
                self.hub.key(self.key_names[e.key],e.type==p.KEYDOWN)
            elif e.type in (p.MOUSEBUTTONDOWN,p.MOUSEBUTTONUP) and e.button in (1,2,3):
                self.hub.button(e.button-1,e.type==p.MOUSEBUTTONDOWN)
            elif e.type==p.MOUSEMOTION:
                self.hub.motion(e.pos,e.rel)
            elif e.type==p.MOUSEWHEEL:
                self.hub.wheel((e.x,e.y))
            elif e.type in (p.WINDOWFOCUSLOST,p.QUIT):
                self.hub.focus(False)
            elif e.type==p.WINDOWFOCUSGAINED:
                self.hub.focus(True)
        self.hub.heartbeat()


class OpenCVCamera:
    """Opt-in camera/video capture, fixed RGB uint8 HWC output.

    Construct/open/read/close on ONE capture thread. read() may block in the
    driver; do not call it from a graph worker/UI thread. A camera process is
    preferable if hard cancellation is required. No device opened on import.
    """
    def __init__(self, *, width=320,height=240,clock=None):
        if type(width) is not int or type(height) is not int or min(width,height)<=0:
            raise ValueError("camera dimensions must be positive integers")
        self.width,self.height = width,height
        self.channel = InputChannel((Port("rgb","1",(height,width,3),"camera",quantity="rgb_uint8",dtype="uint8"),),
                                    **({"clock":clock} if clock is not None else {}))
        self._capture,self._cv,self._owner = None,None,None

    def open(self,device=0, *, cv2_module=None):
        if self._capture is not None:
            raise RuntimeError("camera already open")
        if cv2_module is None:
            import cv2 as cv2_module
        self._owner,self._cv = get_ident(),cv2_module
        cap = cv2_module.VideoCapture(device)
        if not cap.isOpened():
            cap.release()
            self.channel.disconnect("camera could not open")
            raise RuntimeError("camera could not open")
        self._capture = cap

    def read(self):
        if self._capture is None or get_ident()!=self._owner:
            raise RuntimeError("camera must be open on its capture thread")
        try:
            ok,bgr = self._capture.read()
            if not ok:
                self.channel.disconnect("camera read failed or end of stream")
                return False
            rgb = self._cv.cvtColor(bgr,self._cv.COLOR_BGR2RGB)
            if rgb.shape!=(self.height,self.width,3):
                rgb = self._cv.resize(rgb,(self.width,self.height))
            self.channel.publish({"rgb":rgb})
            return True
        except BaseException:
            self.channel.disconnect("camera read exception")
            raise

    def close(self):
        if self._capture is not None:
            if get_ident()!=self._owner:
                raise RuntimeError("camera must close on capture thread")
            self._capture.release()
            self._capture = None
        self.channel.disconnect("camera closed")
