import os
import subprocess
import re
import csv
import sys

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "config"))
TEMP_CONFIG_NAME = "autogen_test"
TEMP_CONFIG_FILE = os.path.join(CONFIG_DIR, f"{TEMP_CONFIG_NAME}.mk")

# Allow selection of simulation app via environment variable or default to fc_layer
SIM_APP = os.environ.get('SIM_APP', 'fc_layer')

# Per-run output file name template
OUTPUT_CSV_TEMPLATE = f"ara_{SIM_APP}_l{{lanes}}_vlen{{vlen_bytes}}.csv"

# Simulation parameters
LANES_LIST = [2, 4]
VLEN_BYTES_LIST = [4096, 8192]

# Use nohup to capture output on lab servers
USE_NOHUP = True
NOHUP_OUT = os.path.join(BASE_DIR, "nohup.out")

# Overwrite CSV per run to avoid mixing results across runs
OVERWRITE_CSV = True

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def write_config_file(lanes, vlen_bits):
    content = f"""# Auto-generated configuration for testing
nr_lanes ?= {lanes}
vlen ?= {vlen_bits}
"""
    try:
        with open(TEMP_CONFIG_FILE, 'w') as f:
            f.write(content)
        return True
    except IOError as e:
        print(f"❌ Error writing config file: {e}")
        return False

def run_command(command, cwd=None):
    """
    Runs a shell command and streams output immediately using stdbuf logic.
    """
    print(f"⚡ Running: {command}")
    
    # We prepend 'stdbuf -oL -eL' to force the C++ simulator
    # to flush its buffer on every new line.
    # -oL = Output Line buffered
    # -eL = Error Line buffered
    unbuffered_command = f"stdbuf -oL -eL {command}"

    process = subprocess.Popen(
        unbuffered_command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True, 
        cwd=cwd
    )

    captured_output = []

    # Read character by character or line by line to ensure real-time streaming
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush() # Force Python to print to terminal immediately
            captured_output.append(line)

    return "".join(captured_output)

def run_command_nohup(command, cwd=None, nohup_out=NOHUP_OUT):
    """
    Runs a shell command via nohup and captures output from nohup.out.
    """
    print(f"⚡ Running (nohup): {command}")

    if os.path.exists(nohup_out):
        os.remove(nohup_out)

    unbuffered_command = f"nohup stdbuf -oL -eL {command} > {nohup_out} 2>&1"

    process = subprocess.Popen(
        unbuffered_command,
        shell=True,
        cwd=cwd
    )

    process.wait()

    if not os.path.exists(nohup_out):
        return ""

    with open(nohup_out, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def parse_results(output_text, lanes, vlen_bytes):
    # Parse custom printf_ cycle lines from fc_layer and conv_layer
    line_pattern = re.compile(
        r'^cycles\[epoch\s+(\d+)\s+batch\s+(\d+)/(\d+)\]:\s+(.+)$',
        re.MULTILINE
    )
    kv_pattern = re.compile(r'([a-zA-Z0-9_+\-]+)=([0-9]+)')
    
    # Parse conv backward breakdown (optional, for conv_layer)
    breakdown_pattern = re.compile(
        r'^conv backward breakdown:\s+(.+)$',
        re.MULTILINE
    )

    parsed_data = []

    for match in line_pattern.finditer(output_text):
        epoch, batch, batch_total, kv_blob = match.groups()
        kvs = dict(kv_pattern.findall(kv_blob))

        # Look for breakdown info on the next line(s)
        breakdown_kvs = {}
        breakdown_match = breakdown_pattern.search(output_text, match.end())
        if breakdown_match:
            breakdown_blob = breakdown_match.group(1)
            breakdown_kvs = dict(kv_pattern.findall(breakdown_blob))

        parsed_data.append({
            'Lanes': lanes,
            'VLEN (Bytes)': vlen_bytes,
            'VLEN (Bits)': vlen_bytes * 8,
            'Epoch': int(epoch),
            'Batch': int(batch),
            'Batch Total': int(batch_total),
            'Prep Cycles': kvs.get('prep', ''),
            'Forward Cycles': kvs.get('forward', ''),
            'Loss Cycles': kvs.get('loss', ''),
            'Zero D Input Cycles': kvs.get('zero_d_input', ''),
            'Backward Cycles': kvs.get('backward', ''),
            'Update Cycles': kvs.get('update', ''),
            'D Input Cycles': breakdown_kvs.get('d_input', kvs.get('backward_d_input', kvs.get('d_input', ''))),
            'D Bias Cycles': breakdown_kvs.get('d_bias', kvs.get('backward_d_bias', kvs.get('d_bias', ''))),
            'D Weights Im2col Cycles': breakdown_kvs.get('d_weights_im2col', ''),
            'D Weights Total Cycles': breakdown_kvs.get('d_weights_total', kvs.get('backward_d_weights', kvs.get('d_weight', ''))),
            'Total Backward Cycles': breakdown_kvs.get('total', '')
        })

    return parsed_data

def append_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = [
        'Lanes',
        'VLEN (Bytes)',
        'VLEN (Bits)',
        'Epoch',
        'Batch',
        'Batch Total',
        'Prep Cycles',
        'Forward Cycles',
        'Loss Cycles',
        'Zero D Input Cycles',
        'Backward Cycles',
        'Update Cycles',
        'D Input Cycles',
        'D Bias Cycles',
        'D Weights Im2col Cycles',
        'D Weights Total Cycles',
        'Total Backward Cycles'
    ]

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

def main():
    print("========================================")
    print("      ARA AUTOMATED BENCHMARK SUITE    ")
    print("========================================")
    print(f"Testing application: {SIM_APP}")
    print("(Set SIM_APP environment variable to change: export SIM_APP=conv_layer)")
    print("========================================\n")

    for lanes in LANES_LIST:
        for vlen_bytes in VLEN_BYTES_LIST:
            vlen_bits = vlen_bytes * 8

            output_csv = os.path.join(
                BASE_DIR,
                OUTPUT_CSV_TEMPLATE.format(lanes=lanes, vlen_bytes=vlen_bytes)
            )

            if OVERWRITE_CSV and os.path.exists(output_csv):
                os.remove(output_csv)
            
            if not write_config_file(lanes, vlen_bits):
                sys.exit(1)

            # 1. Run Verilate
            run_command(f"make verilate config={TEMP_CONFIG_NAME}")
            
            print(f"\n\n>>> 🚀 STARTING TEST [{SIM_APP}]: Lanes={lanes}, VLEN={vlen_bytes}B ({vlen_bits} bits) <<<")

            # 2. Run Simulation
            sim_cmd = f"make simv app={SIM_APP} config={TEMP_CONFIG_NAME}"
            if USE_NOHUP:
                sim_output = run_command_nohup(sim_cmd)
            else:
                sim_output = run_command(sim_cmd)

            # 3. Parse and Save
            metrics = parse_results(sim_output, lanes, vlen_bytes)

            if metrics:
                print(f"✅ Found {len(metrics)} data points. Saving to CSV...")
                append_to_csv(metrics, output_csv)
            else:
                print("⚠️  No metrics found. (If this is the first run, ensure output is appearing)")

    print("\n\n========================================")
    print(f"🎉 ALL TESTS COMPLETED.")
    print(f"📊 Results saved in: {os.path.abspath(BASE_DIR)}")
    print("========================================")

    if os.path.exists(TEMP_CONFIG_FILE):
        os.remove(TEMP_CONFIG_FILE)

if __name__ == "__main__":
    main()