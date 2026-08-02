"""
fpga_driver.py
--------------
UART driver for communicating between the PC and the Basys 3 FPGA.

Protocol summary:
  PC → FPGA:  [0xAA] [TYPE] [N data bytes] [XOR checksum] [0x55]
  FPGA → PC:  [0xAA] [2048 data bytes] [XOR checksum]
  ACK:        [0x06]  (weight-load frames only)

The two directions are deliberately asymmetric:

  * The FPGA does NOT verify the checksum or stop byte it receives -- it
    counts payload bytes and moves on, swallowing the trailing two bytes.
    They are still sent so the frame stays self-describing on a scope/logic
    analyser, but they carry no authority. There is no NAK.

  * The FPGA's reply carries no TYPE byte and no stop byte. The PC only ever
    receives one kind of frame (an attention result), so there is nothing to
    disambiguate. The PC does verify the checksum it receives.

  * ACK (0x06) is sent for weight-load frames only. Live inference frames are
    NOT acknowledged -- the result frame is itself the acknowledgement. Do not
    wait for an ACK after sending an inference packet; the first byte back
    will be the 0xAA that opens the result frame.

TYPE byte:
  Live inference packets (16x64, 2048-byte payload):
    0x01 = Layer 1 embedding  (FPGA uses W_Q1, W_K1, W_V1)
    0x02 = Layer 2 embedding  (FPGA uses W_Q2, W_K2, W_V2)

  Weight-load packets (64x64, 8192-byte payload), sent once at startup
  before any live inference packets. The FPGA infers payload size from
  TYPE rather than assuming a fixed 2048-byte length. There is no
  weight-loading "mode" on the FPGA -- every frame is dispatched purely
  on its TYPE byte -- but the weights must still all be loaded before the
  first inference, since nothing else initialises weight BRAM:
    0x10 = W_q1
    0x11 = W_k1
    0x12 = W_v1
    0x13 = W_out1
    0x14 = W_q2
    0x15 = W_k2
    0x16 = W_v2
    0x17 = W_out2

Data format:
  Live packets:   16x64 = 1024 values  -> 2048 bytes per packet.
  Weight packets: 64x64 = 4096 values  -> 8192 bytes per packet.
  All values are Q8.8 fixed-point (signed 16-bit integer), sent
  row-major, big-endian.
"""

import sys
import time
import struct
from typing import Optional

import serial
import numpy as np

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

BAUD_RATE   = 921600
MATRIX_ROWS = 16
MATRIX_COLS = 64
DATA_BYTES  = MATRIX_ROWS * MATRIX_COLS * 2   # 2048 bytes (Q8.8 = 2 bytes/value)

START_BYTE  = 0xAA
STOP_BYTE   = 0x55   # sent by the PC, never received -- the FPGA's reply has no stop byte
ACK_BYTE    = 0x06   # weight-load frames only; inference frames are not ACKed

LAYER_1     = 0x01
LAYER_2     = 0x02

WEIGHT_ROWS = 64
WEIGHT_COLS = 64
WEIGHT_DATA_BYTES = WEIGHT_ROWS * WEIGHT_COLS * 2   # 8192 bytes (Q8.8 = 2 bytes/value)

# TYPE bytes for the 8 weight matrices, sent once at startup. The low 3 bits are
# the matrix index the FPGA uses to pick a 4096-word weight BRAM region, so the
# values matter even though the send order does not.
WEIGHT_TYPE = {
    'W_q1':   0x10,
    'W_k1':   0x11,
    'W_v1':   0x12,
    'W_out1': 0x13,
    'W_q2':   0x14,
    'W_k2':   0x15,
    'W_v2':   0x16,
    'W_out2': 0x17,
}
WEIGHT_ORDER = ['W_q1', 'W_k1', 'W_v1', 'W_out1', 'W_q2', 'W_k2', 'W_v2', 'W_out2']

MAX_RETRIES = 3
TIMEOUT_SEC = 2.0   # seconds to wait for FPGA response

# Q8.8 value range as floats: signed 16-bit integer / 256
Q8_8_MIN = -128.0       # -32768 / 256
Q8_8_MAX =  127.99609   # +32767 / 256


# ─────────────────────────────────────────────
#  Q8.8 CONVERSION
# ─────────────────────────────────────────────

def float_to_q8_8(matrix: np.ndarray) -> bytes:
    """
    Convert a float32 numpy matrix to Q8.8 fixed-point bytes.

    Q8.8 means: 1 sign bit, 7 integer bits, 8 fractional bits.
    Range: -128.0 to +127.996
    Precision: 1/256 ~ 0.004

    Each value becomes a signed 16-bit integer, sent big-endian.
    Example: 1.5 -> round(1.5 * 256) = 384 = 0x0180

    IMPORTANT: We clamp as float BEFORE casting to int16.
    Casting out-of-range floats directly to int16 causes silent
    wraparound overflow in numpy (e.g. 200.0 -> garbage).
    """
    flat = matrix.flatten().astype(np.float32)

    # Clamp to Q8.8 representable range FIRST (as float)
    flat = np.clip(flat, Q8_8_MIN, Q8_8_MAX)

    # Scale and round, then cast to int16 -- safe now that values are clamped
    scaled = np.round(flat * 256.0).astype(np.int16)

    # Pack as big-endian signed 16-bit integers
    return struct.pack(f'>{len(scaled)}h', *scaled)


def q8_8_to_float(data: bytes) -> np.ndarray:
    """
    Convert Q8.8 fixed-point bytes back to a float32 numpy matrix.

    Reverses float_to_q8_8: unpack big-endian int16s, divide by 256.
    Returns a (MATRIX_ROWS x MATRIX_COLS) float32 array.
    """
    num_values = len(data) // 2
    integers = struct.unpack(f'>{num_values}h', data)
    floats = np.array(integers, dtype=np.float32) / 256.0
    return floats.reshape(MATRIX_ROWS, MATRIX_COLS)


# ─────────────────────────────────────────────
#  PACKET BUILDING
# ─────────────────────────────────────────────

def compute_checksum(data: bytes) -> int:
    """
    XOR all bytes in data together to produce a 1-byte checksum.
    Simple, fast, and easy to implement in Verilog on the FPGA side.
    """
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum


def build_packet(type_byte: int, matrix: np.ndarray) -> bytes:
    """
    Build a full UART packet for sending a matrix to the FPGA.

    Payload size follows from the matrix's own shape, so this works for
    both live inference packets (16x64 -> 2048 data bytes) and weight-load
    packets (64x64 -> 8192 data bytes).

    Packet layout:
      [0xAA]        1 byte  - start marker
      [TYPE]        1 byte  - see module docstring for the TYPE byte table
      [DATA]     N bytes    - Q8.8 encoded matrix, row-major, big-endian
      [CHECKSUM]    1 byte  - XOR of all N data bytes only
      [0x55]        1 byte  - stop marker
    """
    data = float_to_q8_8(matrix)
    checksum = compute_checksum(data)
    return bytes([START_BYTE, type_byte]) + data + bytes([checksum, STOP_BYTE])


# ─────────────────────────────────────────────
#  FPGA DRIVER CLASS
# ─────────────────────────────────────────────

class FPGADriver:
    """
    Manages the serial connection to the Basys 3 FPGA and handles
    sending/receiving embedding matrices.

    Usage:
        driver = FPGADriver(port='COM3')           # Windows
        driver = FPGADriver(port='/dev/ttyUSB0')   # Linux/Mac

        # As a context manager (recommended -- auto disconnects):
        with FPGADriver(port='COM3') as driver:
            result = driver.run_inference(embedding_matrix)

        # Or manually:
        driver.connect()
        result = driver.run_inference(embedding_matrix)
        driver.disconnect()
    """

    def __init__(self, port: str, baud_rate: int = BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.ser: Optional[serial.Serial] = None

    def connect(self):
        """Open the serial port to the FPGA."""
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT_SEC
        )
        print(f"[FPGA] Connected on {self.port} at {self.baud_rate} baud")

    def disconnect(self):
        """Close the serial port."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[FPGA] Disconnected")

    def __enter__(self) -> "FPGADriver":
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    # ── Low-level send/receive ──────────────────

    def _send_packet(self, layer: int, matrix: np.ndarray):
        """
        Send one live inference packet to the FPGA.

        There is no reply to wait for here: the FPGA does not ACK inference
        frames. It goes straight from the last payload byte into the matmul
        chain, and the next byte on the wire is the 0xAA that opens the
        result frame. Reading a byte here would steal that 0xAA and desync
        the receive. Call _receive_packet() next.
        """
        self.ser.write(build_packet(layer, matrix))

    def _send_weight_packet(self, type_byte: int, matrix: np.ndarray) -> bool:
        """
        Send one weight matrix packet to the FPGA and wait for ACK.

        Weight frames are the only ones the FPGA acknowledges, and a weight
        load gets no result matrix back -- just the one ACK byte.

        Note the ACK is emitted on the final payload byte, before the FPGA
        has even seen the checksum, and the FPGA does not verify checksums.
        So an ACK means "8192 bytes were counted in", not "the data is good".
        Returns True on ACK, False on timeout or any other byte.
        """
        packet = build_packet(type_byte, matrix)
        self.ser.write(packet)

        response = self.ser.read(1)
        if not response:
            print(f"[FPGA] Timeout waiting for ACK (weight type 0x{type_byte:02X})")
            return False
        if response[0] == ACK_BYTE:
            return True
        print(f"[FPGA] Unexpected response byte: 0x{response[0]:02X} "
              f"(weight type 0x{type_byte:02X})")
        return False

    def _receive_packet(self) -> Optional[np.ndarray]:
        """
        Receive a result packet from the FPGA and return it as a float matrix.

        Reads: [0xAA] [2048 bytes] [CHECKSUM]

        No TYPE byte and no stop byte -- the FPGA only ever sends this one
        frame shape, so there is nothing to disambiguate and nothing after
        the checksum. Returns None if anything goes wrong.

        The first read blocks for the FPGA's whole compute time (all seven
        matmuls plus softmax), so TIMEOUT_SEC has to cover that, not just
        the wire time.
        """
        # Read start byte
        start = self.ser.read(1)
        if not start or start[0] != START_BYTE:
            print(f"[FPGA] Bad start byte: {start.hex() if start else 'timeout'}")
            return None

        # Read the 2048 data bytes
        data = self.ser.read(DATA_BYTES)
        if len(data) != DATA_BYTES:
            print(f"[FPGA] Expected {DATA_BYTES} data bytes, got {len(data)}")
            return None

        # Read the trailing checksum
        trailer = self.ser.read(1)
        if len(trailer) != 1:
            print("[FPGA] Timeout reading checksum")
            return None

        received_checksum = trailer[0]
        expected_checksum = compute_checksum(data)
        if received_checksum != expected_checksum:
            print(f"[FPGA] Checksum mismatch: "
                  f"received 0x{received_checksum:02X}, "
                  f"expected 0x{expected_checksum:02X}")
            return None

        return q8_8_to_float(data)

    # ── High-level exchange with retry ─────────

    def _exchange(self, layer: int, matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Send a matrix to the FPGA and receive the attention result.
        Retries up to MAX_RETRIES times on any failure.

        The input buffer is flushed before every attempt, not just retries.
        With no ACK on inference frames there is no handshake to resync on:
        a single stale byte sitting in the buffer would be mistaken for the
        result frame's start byte and desync every subsequent read.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                print(f"[FPGA] Retry {attempt}/{MAX_RETRIES}...")

            self.ser.reset_input_buffer()
            self._send_packet(layer, matrix)

            result = self._receive_packet()
            if result is not None:
                return result

        print(f"[FPGA] Failed after {MAX_RETRIES} attempts on layer 0x{layer:02X}")
        return None

    # ── Public API ──────────────────────────────

    def load_weights(self, weights: dict) -> bool:
        """
        Load all 8 weight matrices into FPGA BRAM. Call once at startup,
        before any run_inference() calls.

        Args:
            weights: dict mapping matrix name -> float32 numpy array of
                     shape (64, 64). Must contain all of WEIGHT_ORDER:
                     'W_q1', 'W_k1', 'W_v1', 'W_out1',
                     'W_q2', 'W_k2', 'W_v2', 'W_out2'.

        Sends matrices in WEIGHT_ORDER, each with its own retry loop.
        The FPGA is expected to leave weight-loading mode once it ACKs
        the last matrix (W_out2, type 0x17).

        Returns:
            True if all 8 matrices were ACKed, False if any failed
            after MAX_RETRIES attempts.
        """
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("[FPGA] Not connected. Call connect() first.")

        missing = [name for name in WEIGHT_ORDER if name not in weights]
        if missing:
            raise ValueError(f"Missing weight matrices: {missing}")

        t0 = time.time()

        for name in WEIGHT_ORDER:
            matrix = weights[name]
            if matrix.shape != (WEIGHT_ROWS, WEIGHT_COLS):
                raise ValueError(
                    f"{name}: expected shape ({WEIGHT_ROWS}, {WEIGHT_COLS}), "
                    f"got {matrix.shape}"
                )

            type_byte = WEIGHT_TYPE[name]
            print(f"[FPGA] Sending {name} (type 0x{type_byte:02X})...")

            ok = False
            for attempt in range(1, MAX_RETRIES + 1):
                if attempt > 1:
                    print(f"[FPGA] Retry {attempt}/{MAX_RETRIES} -- flushing buffer...")
                    self.ser.reset_input_buffer()
                if self._send_weight_packet(type_byte, matrix):
                    ok = True
                    break

            if not ok:
                print(f"[FPGA] Failed to load {name} after {MAX_RETRIES} attempts")
                return False

        elapsed_ms = (time.time() - t0) * 1000
        print(f"[FPGA] All 8 weight matrices loaded in {elapsed_ms:.1f} ms")
        return True

    def run_inference(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        """
        Run a full 2-layer attention inference on the FPGA.

        Args:
            embedding: float32 numpy array of shape (16, 64).
                       The embedded token matrix produced by pixel_completer.py.

        Returns:
            float32 numpy array of shape (16, 64) -- the result after both
            attention layers have been applied by the FPGA.
            Returns None if communication fails after all retries.
        """
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("[FPGA] Not connected. Call connect() first.")

        if embedding.shape != (MATRIX_ROWS, MATRIX_COLS):
            raise ValueError(
                f"Expected embedding shape ({MATRIX_ROWS}, {MATRIX_COLS}), "
                f"got {embedding.shape}"
            )

        t0 = time.time()

        print("[FPGA] Sending Layer 1 embedding...")
        result_l1 = self._exchange(LAYER_1, embedding)
        if result_l1 is None:
            return None

        print("[FPGA] Sending Layer 2 embedding...")
        result_l2 = self._exchange(LAYER_2, result_l1)
        if result_l2 is None:
            return None

        elapsed_ms = (time.time() - t0) * 1000
        print(f"[FPGA] Inference complete in {elapsed_ms:.1f} ms")
        return result_l2


# ─────────────────────────────────────────────
#  SELF-TEST  (python fpga_driver.py [PORT])
# ─────────────────────────────────────────────

def run_conversion_test() -> bool:
    """
    Test Q8.8 encode/decode round-trip with no hardware required.
    Returns True if max error is within spec.
    """
    print("--- Q8.8 round-trip test ---")

    # Test normal range
    test = np.random.uniform(-5.0, 5.0, (MATRIX_ROWS, MATRIX_COLS)).astype(np.float32)
    encoded = float_to_q8_8(test)
    decoded = q8_8_to_float(encoded)
    max_err = float(np.max(np.abs(test - decoded)))
    print(f"  Normal range (-5 to 5):  max error = {max_err:.6f}  (limit: 0.004)")

    # Test clamping -- values outside Q8.8 range must not crash or overflow
    extreme = np.full((MATRIX_ROWS, MATRIX_COLS), 200.0, dtype=np.float32)
    decoded_extreme = q8_8_to_float(float_to_q8_8(extreme))
    clamped_val = float(decoded_extreme[0, 0])
    print(f"  Overflow clamp (input=200.0): got {clamped_val:.5f}  (expected ~127.996)")

    # Test packet structure
    dummy = np.zeros((MATRIX_ROWS, MATRIX_COLS), dtype=np.float32)
    pkt = build_packet(LAYER_1, dummy)
    assert pkt[0]  == START_BYTE,      "Start byte wrong"
    assert pkt[1]  == LAYER_1,         "Type byte wrong"
    assert pkt[-1] == STOP_BYTE,       "Stop byte wrong"
    assert len(pkt) == DATA_BYTES + 4, f"Packet length wrong: {len(pkt)}"
    print(f"  Packet structure: OK ({len(pkt)} bytes)")

    passed = max_err < 0.004
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"

    print("=" * 40)
    print("  FPGA UART Driver Self-Test")
    print("=" * 40)
    print(f"Port: {port}  |  Baud: {BAUD_RATE}")
    print()

    # Always run the conversion test (no hardware needed)
    passed = run_conversion_test()
    print()

    if not passed:
        print("Conversion test FAILED -- do not proceed to hardware test.")
        sys.exit(1)

    # Hardware test -- only meaningful if FPGA is connected
    print(f"--- Hardware test on {port} ---")
    try:
        with FPGADriver(port=port) as driver:
            fake_embedding = np.random.uniform(
                -1.0, 1.0, (MATRIX_ROWS, MATRIX_COLS)
            ).astype(np.float32)
            print(f"Sending fake embedding (shape {fake_embedding.shape})...")
            result = driver.run_inference(fake_embedding)

            if result is not None:
                print(f"Received result shape : {result.shape}")
                print(f"First row of result   : {np.round(result[0], 3)}")
            else:
                print("Inference returned None -- check FPGA firmware.")

    except serial.SerialException as e:
        print(f"Could not open {port}: {e}")
        print("(Expected if FPGA is not connected yet -- conversion test still passed.)")