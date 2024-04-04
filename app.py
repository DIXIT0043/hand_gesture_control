import tkinter as tk
import subprocess
import os

def execute_HGM():
    # Get the directory of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the HGM.py file
    hgm_path = os.path.join(current_directory, "HGM.py")
    # Execute the HGM.py file using subprocess
    subprocess.Popen(["python", hgm_path])

# Create the main application window
root = tk.Tk()
root.title("Mouse Control App")

# Create a label
label = tk.Label(root, text="Let's Control Your Mouse with Magic", padx=10, pady=10)
label.pack()

# Function to execute HGM.py when the button is clicked
button = tk.Button(root, text="Let's Control Your Mouse with Magic", command=execute_HGM)
button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
