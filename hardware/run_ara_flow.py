import os
import subprocess
import re
import csv
import sys

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_CSV = "ara_benchmark_ideal_dispatcher.csv"
CONFIG_DIR = "../config"
TEMP_CONFIG_NAME = "autogen_test"
TEMP_CONFIG_FILE = os.path.join(CONFIG_DIR, f"{TEMP_CONFIG_NAME}.mk")

# Simulation parameters
LANES_LIST = [2, 4, 8, 16]
VLEN_BYTES_LIST = [32, 64, 128, 256, 512, 1024] 

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
        text=True, 
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

def parse_results(output_text, lanes, vlen_bytes):
    # Regex to capture metrics
    pattern = re.compile(
        r'Calculating a \((\d+) x \1\) x \(\1 x \1\) matrix multiplication\.\.\.'
        r'.*?'
        r'The execution took (\d+) cycles\.'
        r'.*?'
        r'The performance is (\d+\.\d+) FLOP/cycle',
        re.DOTALL | re.MULTILINE
    )

    matches = pattern.findall(output_text)
    parsed_data = []

    for matrix_n, cycles, flop_per_cycle in matches:
        parsed_data.append({
            'Lanes': lanes,
            'VLEN (Bytes)': vlen_bytes,
            'VLEN (Bits)': vlen_bytes * 8,
            'Matrix Size (N)': matrix_n,
            'Cycles': cycles,
            'FLOP/cycle': flop_per_cycle
        })
    
    return parsed_data

def append_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = ['Lanes', 'VLEN (Bytes)', 'VLEN (Bits)', 'Matrix Size (N)', 'Cycles', 'FLOP/cycle']

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
    print("      ARA AUTOMATED BENCHMARK SUITE WITH IDEAL DISPATCHER    ")
    print("========================================")

    for lanes in LANES_LIST:
        for vlen_bytes in VLEN_BYTES_LIST:
            vlen_bits = vlen_bytes * 8
            
           
            
            if not write_config_file(lanes, vlen_bits):
                sys.exit(1)

            # 1. Run Verilate
            run_command(f"make verilate config={TEMP_CONFIG_NAME} ideal_dispatcher=1")
            
            print(f"\n\n>>> 🚀 STARTING TEST: Lanes={lanes}, VLEN={vlen_bytes}B ({vlen_bits} bits) <<<")

            # 2. Run Simulation (Note: we pass app variable differently to handle stdbuf cleanly)
            # We use 'make simv app=fmatmul' so stdbuf applies to the make command
            sim_cmd = "make simv app=fmatmul ideal_dispatcher=1" 
            sim_output = run_command(sim_cmd)

            # 3. Parse and Save
            metrics = parse_results(sim_output, lanes, vlen_bytes)

            if metrics:
                print(f"✅ Found {len(metrics)} data points. Saving to CSV...")
                append_to_csv(metrics, OUTPUT_CSV)
            else:
                print("⚠️  No metrics found. (If this is the first run, ensure output is appearing)")

    print("\n\n========================================")
    print(f"🎉 ALL TESTS COMPLETED.")
    print(f"📊 Results saved in: {os.path.abspath(OUTPUT_CSV)}")
    print("========================================")

    if os.path.exists(TEMP_CONFIG_FILE):
        os.remove(TEMP_CONFIG_FILE)

if __name__ == "__main__":
    main()