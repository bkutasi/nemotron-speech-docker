#!/usr/bin/env python3
"""Always-on VAD dictation daemon (stdlib only).

Captures the tunneled laptop mic continuously (PULSE_SERVER=tcp:localhost:4713),
segments utterances on silence via RMS energy VAD, POSTs each segment to the
local Parakeet sidecar (http://localhost:3005/transcribe), and types the text
into the omp pane via `tmux send-keys -l` (literal, no Enter).

Usage:
  PULSE_SERVER=tcp:localhost:4713 python3 vad_dictate.py --target omp:0.0
  python3 vad_dictate.py --check            # preflight: parecord, sidecar, tmux target
  python3 vad_dictate.py --selftest         # offline VAD segmenter check, no mic/net
  python3 vad_dictate.py --transcribe-file /tmp/utt.wav   # POST path + latency only

Pause: touch ~/.vad-paused  (daemon drops segments while present).
Voice toggles: saying "dictation pause" pauses, "dictation resume" resumes.
F1 interplay: daemon cannot detect omp's F1 recorder — pause the daemon
(voice or pause file) before using F1 or text interleaves.
"""
import argparse
import io
import math
import os
import statistics
import struct
import subprocess
import sys
import time
import urllib.request
import wave
from collections import deque

SR = 16000
FRAME_MS = 20                       # faster silence detection; 640 bytes s16 mono
FRAME_BYTES = SR * 2 * FRAME_MS // 1000
CALIBRATE_S = 2.0                   # noise-floor sample window on (re)start
SILENCE_HANGOVER_S = 0.32           # 320ms end-of-speech; shorter pauses may split sentences
MIN_UTT_S = 0.4                     # shorter -> dropped (sidecar guards <0.25s)
MAX_UTT_S = 30.0                    # force-flush ceiling
PREROLL_S = 0.2                     # preserve speech onsets without excess audio
BACKOFFS = (5, 15, 30)              # tunnel-drop retry delays, last repeats
PAUSE_FILE = os.path.expanduser("~/.vad-paused")
SIDECAR = "http://localhost:3005/transcribe"
PAUSE_PHRASE = "dictation pause"
RESUME_PHRASE = "dictation resume"


def log(msg, logf):
    line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if logf:
        logf.write(line + "\n")
        logf.flush()


def frame_rms(frame):
    n = len(frame) // 2
    if n == 0:
        return 0.0
    vals = struct.unpack("<%dh" % n, frame[: n * 2])
    return math.sqrt(sum(v * v for v in vals) / n)


class Segmenter:
    """RMS energy VAD: calibrate() on noise, then feed() frames -> completed utterances."""

    def __init__(self, floor_rms):
        self.thresh = max(floor_rms * 3.0, 150.0)
        sil_frames = int(SILENCE_HANGOVER_S * 1000 / FRAME_MS)
        self.max_sil = sil_frames
        self.min_frames = int(MIN_UTT_S * 1000 / FRAME_MS)
        self.max_frames = int(MAX_UTT_S * 1000 / FRAME_MS)
        self.preroll = deque(maxlen=max(1, int(PREROLL_S * 1000 / FRAME_MS)))
        self.buf = None
        self.sil = 0

    def feed(self, frame):
        """Return list of frames when an utterance completes, else None."""
        speech = frame_rms(frame) >= self.thresh
        if self.buf is None:
            self.preroll.append(frame)
            if speech:
                self.buf = list(self.preroll)
                self.preroll.clear()
                self.sil = 0
            return None
        self.buf.append(frame)
        if len(self.buf) >= self.max_frames:  # ceiling: flush, fresh search after
            utt = self.buf
            self.buf = None
            self.sil = 0
            return utt
        if speech:
            self.sil = 0
        else:
            self.sil += 1
            if self.sil >= self.max_sil:
                utt = self.buf
                self.buf = None
                self.sil = 0
                # strip trailing silence frames from the segment
                utt = utt[: len(utt) - self.max_sil] or utt
                if len(utt) >= self.min_frames:
                    return utt
        return None


def to_wav(pcm_frames):
    raw = b"".join(pcm_frames)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(raw)
    return bio.getvalue(), len(raw) / 2 / SR


def transcribe(wav_bytes, sidecar, timeout=60):
    boundary = "----vaddictate%x" % int(time.time() * 1000)
    crlf = b"\r\n"
    body = (
        b"--" + boundary.encode() + crlf
        + b'Content-Disposition: form-data; name="language"' + crlf + crlf
        + b"en" + crlf
        + b"--" + boundary.encode() + crlf
        + b'Content-Disposition: form-data; name="file"; filename="utt.wav"' + crlf
        + b"Content-Type: audio/wav" + crlf + crlf
        + wav_bytes + crlf
        + b"--" + boundary.encode() + b"--" + crlf
    )
    req = urllib.request.Request(
        sidecar, data=body,
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        import json
        resp = json.loads(r.read().decode())
    return resp.get("text", "").strip(), time.perf_counter() - t0


def send_to_tmux(text, target):
    try:
        subprocess.run(["tmux", "send-keys", "-t", target, "-l", text],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if "can't find pane" in (e.stderr or ""):
            raise SystemExit(2)  # target gone -> supervisor re-resolves
        raise


def read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None if not buf else buf  # None = clean EOF
        buf += chunk
    return buf


def capture_loop(args, logf):
    backoff_i = 0
    while True:
        proc = subprocess.Popen(
            ["parecord", "--device=" + args.device, "--format=s16le",
             "--rate=16000", "--channels=1", "--file-format=raw",
             "--latency-msec=40", "--process-time-msec=20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={**os.environ, "PULSE_SERVER": os.environ.get("PULSE_SERVER", "tcp:localhost:4713")},
        )
        try:
            log("capture started (device=%s), calibrating %.0fs noise floor..."
                % (args.device, CALIBRATE_S), logf)
            cal = []
            t_end = time.time() + CALIBRATE_S
            while time.time() < t_end:
                f = read_exact(proc.stdout, FRAME_BYTES)
                if f is None:
                    raise EOFError("parecord exited during calibration")
                cal.append(frame_rms(f))
            floor = statistics.median(cal)
            seg = Segmenter(floor)
            log("noise floor=%.0f rms -> speech thresh=%.0f rms" % (floor, seg.thresh), logf)
            backoff_i = 0
            while True:
                f = read_exact(proc.stdout, FRAME_BYTES)
                if f is None or len(f) < FRAME_BYTES:
                    raise EOFError("parecord exited (tunnel drop?)")
                utt = seg.feed(f)
                if utt:
                    handle_utterance(utt, args, logf)
        except EOFError as e:
            delay = BACKOFFS[min(backoff_i, len(BACKOFFS) - 1)]
            backoff_i += 1
            log("capture lost: %s -- retry in %ds" % (e, delay), logf)
            time.sleep(delay)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def handle_utterance(utt, args, logf):
    wav, dur = to_wav(utt)
    t_end = time.perf_counter()
    try:
        text, post_s = transcribe(wav, args.sidecar)
    except Exception as e:
        log("POST %0.1fs audio failed: %s (segment dropped)" % (dur, e), logf)
        return
    low = text.lower()
    if PAUSE_PHRASE in low:
        open(PAUSE_FILE, "w").write("paused by voice at %s\n" % time.ctime())
        log("voice toggle: PAUSED (heard %r)" % text, logf)
        return
    if RESUME_PHRASE in low:
        try:
            os.unlink(PAUSE_FILE)
        except FileNotFoundError:
            pass
        log("voice toggle: RESUMED (heard %r)" % text, logf)
        return
    if not text:
        log("segment %0.1fs -> empty text, skipped (asr %.2fs)" % (dur, post_s), logf)
        return
    if os.path.exists(PAUSE_FILE):
        log("segment %0.1fs dropped (paused): %r" % (dur, text[:60]), logf)
        return
    try:
        send_to_tmux(text, args.target)
        dt = time.perf_counter() - t_end
        log("utt %0.1fs -> %r (asr %.2fs, end-to-type %.2fs)" % (dur, text, post_s, dt), logf)
    except Exception as e:
        log("send-keys to %s failed: %s (text kept in log): %r" % (args.target, e, text), logf)


def cmd_check(args):
    ok = True
    for bin_ in ("parecord", "tmux"):
        found = any(os.access(os.path.join(p, bin_), os.X_OK)
                    for p in os.environ.get("PATH", "").split(os.pathsep))
        print(("OK  " if found else "FAIL") + " binary: " + bin_)
        ok &= found
    try:  # sidecar /ready (urllib, short timeout)
        with urllib.request.urlopen("http://localhost:3005/ready", timeout=5) as r:
            print("OK   sidecar /ready:", r.read().decode()[:100])
    except Exception as e:
        print("FAIL sidecar /ready:", e)
        ok = False
    r = subprocess.run(["tmux", "display-message", "-t", args.target, "-p", "#{pane_id}"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("OK   tmux target %s -> pane %s" % (args.target, r.stdout.strip()))
    else:
        print("FAIL tmux target %s: %s" % (args.target, r.stderr.strip()[-120:]))
        ok = False
    print("pause file:", PAUSE_FILE, "(present)" if os.path.exists(PAUSE_FILE) else "(absent)")
    return 0 if ok else 1


def cmd_selftest():
    import array
    seg = Segmenter(50.0)
    got = []
    def feed(sec, amp):
        n = int(sec * 1000 / FRAME_MS)
        for i in range(n):
            if amp:
                # 440 Hz sine, amp RMS ~ amp/sqrt(2); one frame of samples
                fr = array.array("h", (int(amp * math.sin(2 * math.pi * 440 * (i * (FRAME_BYTES // 2) + k) / SR))
                                       for k in range(FRAME_BYTES // 2))).tobytes()
            else:
                fr = b"\x00" * FRAME_BYTES
            r = seg.feed(fr)
            if r:
                got.append(sum(len(x) for x in r) / 2 / SR)
    feed(0.5, 0); feed(1.0, 9000); feed(1.5, 0)
    assert len(got) == 1, "expected 1 utterance, got %d" % len(got)
    assert 0.9 <= got[0] <= 1.5, "utt len %.2fs out of range" % got[0]
    # short blip (<0.4s) must not emit
    seg2 = Segmenter(50.0)
    emitted = [seg2.feed(array.array("h", (9000,) * (FRAME_BYTES // 2)).tobytes()) for _ in range(2)]
    emitted += [seg2.feed(b"\x00" * FRAME_BYTES) for _ in range(20)]
    assert not any(emitted), "short blip must be dropped, got %r" % emitted
    print("selftest OK: 1.0s tone -> %.2fs segment; 0.04s blip dropped" % got[0])


def main():
    ap = argparse.ArgumentParser(description="Always-on VAD dictation daemon")
    ap.add_argument("--target", default=os.environ.get("VAD_TMUX_TARGET", ""),
                    help="tmux pane for omp (e.g. omp:0.0). Env VAD_TMUX_TARGET.")
    ap.add_argument("--device", default="@DEFAULT_SOURCE@")
    ap.add_argument("--sidecar", default=SIDECAR)
    ap.add_argument("--log", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "vad-dictate.log"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--transcribe-file", metavar="WAV")
    args = ap.parse_args()
    if args.selftest:
        return cmd_selftest()
    if args.check:
        if not args.target:
            print("note: --target unset; capture-pane delivery needs it")
            args.target = "0:0.0"
        return cmd_check(args)
    if args.transcribe_file:
        with open(args.transcribe_file, "rb") as f:
            wav = f.read()
        text, post_s = transcribe(wav, args.sidecar)
        print("text=%r asr=%.2fs" % (text, post_s))
        return 0
    if not args.target:
        sys.exit("error: --target or VAD_TMUX_TARGET required (omp pane, e.g. 0:0.0)")
    logf = open(args.log, "a", buffering=1)
    log("vad_dictate start -> tmux %s (pause file %s)" % (args.target, PAUSE_FILE), logf)
    try:
        capture_loop(args, logf)
    except KeyboardInterrupt:
        log("stopped by user", logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
