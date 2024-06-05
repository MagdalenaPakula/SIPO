import tkinter as tk
from tkinter import ttk
import time

# Load model
# import trained_model.
#
# model = trained_model.load_model()


def process_text():
    text = text_input.get("1.0", "end-1c")
    if not text.strip():
        result_label.config(text="Please enter some text.")
        return

    result_label.config(text="Processing, please wait...")
    root.update()

    # sentiment, reason = model.classify(text)

    # if sentiment == "positive":
    #     result_label.config(text=f"The text is positive.\nReason: {reason}")
    # else:
    #     result_label.config(text=f"The text is negative.\nReason: {reason}")


def clear_text():
    text_input.delete("1.0", tk.END)
    result_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("GUI | Review analysis")

    text_input = tk.Text(root, height=5, width=50)
    text_input.grid(row=0, column=0, padx=10, pady=10)

    process_button = tk.Button(root, text="PROCESS", command=process_text, fg='green')
    process_button.grid(row=0, column=1, padx=10, pady=10)

    result_label = tk.Label(root, text="", wraplength=400)
    result_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    clear_button = tk.Button(root, text="Clear", command=clear_text, fg='red')
    clear_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    # Run the main loop
    root.mainloop()