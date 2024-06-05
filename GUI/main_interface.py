import tkinter as tk


def create_window():
    window = tk.Tk()
    window.title("Test interface")
    window.geometry("300x150")

    text_input = tk.Entry(window, width=30)
    text_input.pack(pady=10)

    button_process = tk.Button(window, text="PROCESS")
    button_process.pack()

    result_label = tk.Label(window, text="", fg="green", font=("Arial", 12))
    result_label.pack(pady=10)

    window.mainloop()


if __name__ == '__main__':
    create_window()
