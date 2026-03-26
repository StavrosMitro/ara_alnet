import os
import subprocess
import re
import csv
import sys

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_CSV = "ara_softmax_cycles.csv"
CONFIG_DIR = "../config"
TEMP_CONFIG_NAME = "autogen_softmax"
TEMP_CONFIG_FILE = os.path.join(CONFIG_DIR, f"{TEMP_CONFIG_NAME}.mk")

# Sweep lists
LANES_LIST = [2, 4, 8, 16]
VLEN_BYTES_LIST = [32, 64, 128, 256, 512, 1024]

# Patterns to pick up softmax results
RE_SCALAR  = re.compile(r"scalar SOFTMAX execution took (\d+) cycles", re.IGNORECASE)
RE_VECTOR  = re.compile(r"vector Softmax execution took (\d+) cycles", re.IGNORECASE)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def write_config_file(lanes, vlen_bits):
    content = f"""# Auto-generated configuration
nr_lanes ?= {lanes}
vlen ?= {vlen_bits}
"""
    with open(TEMP_CONFIG_FILE, "w") as f:
        f.write(content)


def run_command(command, cwd=None):
    print(f"⚡ Running: {command}")
    full_cmd = f"stdbuf -oL -eL {command}"

    proc = subprocess.Popen(
        full_cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )

    out_lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_lines.append(line)

    return "".join(out_lines)


def parse_softmax(output):
    """Extract scalar and vector cycles from Ara softmax printout."""
    scalar = RE_SCALAR.search(output)
    vector = RE_VECTOR.search(output)

    return (
        int(scalar.group(1)) if scalar else None,
        int(vector.group(1)) if vector else None
    )


def append_to_csv(row):
    file_exists = os.path.isfile(OUTPUT_CSV)
    headers = ["Lanes", "VLEN (Bytes)", "VLEN (Bits)", "Scalar Cycles", "Vector Cycles"]

    with open(OUTPUT_CSV, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ==========================================
# MAIN
# ==========================================

def main():
    print("========================================")
    print("        ARA SOFTMAX AUTOMATION          ")
    print("========================================")

    for lanes in LANES_LIST:
        for vlen_bytes in VLEN_BYTES_LIST:
            vlen_bits = vlen_bytes * 8

            print(f"\n\n>>> 🚀 Testing Lanes={lanes}, VLEN={vlen_bytes}B ({vlen_bits} bits)")

            write_config_file(lanes, vlen_bits)

            run_command(f"make verilate config={TEMP_CONFIG_NAME}")

            sim_output = run_command("make simv app=softmax")

            scalar, vector = parse_softmax(sim_output)

            if scalar is None and vector is None:
                print("⚠️  DID NOT FIND SOFTMAX METRICS\n")
                continue

            print(f"   → Scalar: {scalar} cycles")
            print(f"   → Vector: {vector} cycles")

            append_to_csv({
                "Lanes": lanes,
                "VLEN (Bytes)": vlen_bytes,
                "VLEN (Bits)": vlen_bits,
                "Scalar Cycles": scalar,
                "Vector Cycles": vector
            })

    print("\n========================================")
    print(f"🎉 Completed sweep. Results saved in {OUTPUT_CSV}")
    print("========================================")

    if os.path.exists(TEMP_CONFIG_FILE):
        os.remove(TEMP_CONFIG_FILE)


if __name__ == "__main__":
    main()
