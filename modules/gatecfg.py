import os
import subprocess

def run_gate(seed):
    gate_file = "./gatemac/detector_setseed.mac"
    seeded_file = f"./gatemac/detector_seed{seed}.mac"
    fin = open(gate_file, "rt")
    fout = open(seeded_file, "wt")
    for line in fin:
        fout.write(line.replace("{seed}", f"{seed}"))
    fin.close()
    fout.close()
    subprocess.run(["/home/arnon/Development/Gate/bin/Gate" ,f"/home/arnon/Projects/telescope-track-following/gatemac/detector_seed{seed}.mac"])
    os.remove(seeded_file)