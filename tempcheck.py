import subprocess
import os
import time
import datetime

if __name__ == "__main__":
    # Command to run in the subprocess
    command = ["watch", "sensors"]
    num_temps = 10

    while True:
        for i in range(num_temps):
            lines = []
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                line = process.stdout.readline()
                lines.append(line)
                if "Adapter: PCI adapter" in line:
                    if i != 0:
                        with open("./heatlogs.txt", 'a') as f:
                            f.write("Datetime: {}".format(datetime.datetime.now().strftime("%c")))
                            f.writelines(lines)
                            f.write("\n#######################################################\n\n")
                    else:
                        with open("./heatlogs.txt", 'w') as f:
                            f.write("Datetime: {}".format(datetime.datetime.now().strftime("%c")))
                            f.writelines(lines)
                            f.write("\n#######################################################\n\n")
                    process.terminate()
                    break
            print("i: {}".format(i))
            time.sleep(5)
                
            # if delta_time > 10:
            #     print("time:-> {}".format((final_time - current_time).total_seconds()))
    try:
        # Read and print stdout and stderr in real-time
        while True:
            # Read a line from stdout
            stdout_line = process.stdout.readline()
            if not stdout_line:
                break  # No more output from stdout

            print(f"stdout: {stdout_line.strip()}")

            # Read a line from stderr
            stderr_line = process.stderr.readline()
            if not stderr_line:
                break  # No more output from stderr

            print(f"stderr: {stderr_line.strip()}")

    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C) to stop the loop gracefully
        pass

    finally:
        # Optionally, you can also get the return code
        return_code = process.poll()
        print(f"Subprocess return code: {return_code}")

        # Close the subprocess if it hasn't finished yet
        process.terminate()
