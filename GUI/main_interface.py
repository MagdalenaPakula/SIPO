import threading
import tkinter as tk
import joblib
import numpy as np

from lime import lime_text

vectorizer = joblib.load('../vectorizer.joblib')
model = joblib.load('../model.joblib')

explainer = lime_text.LimeTextExplainer(class_names=["negative", "positive"])


# (we dont have predict_proba in model creation) aka 'obejście problemu' bo trenowanie nowego modelu zajmuje dłużej
def predict_proba_wrapper(texts):
    # texts to vectors
    transformed_text = vectorizer.transform(texts)

    raw_predictions = model.predict(transformed_text)

    # dummy probabilities
    probabilities = []
    for prediction in raw_predictions:
        if prediction == "positive":
            probabilities.append([0.1, 0.9])  # high probability for positive class
        else:
            probabilities.append([0.9, 0.1])  # high probability for negative class

    return np.array(probabilities)


def explain_instance(text):
    explanation = explainer.explain_instance(text, predict_proba_wrapper, num_features=10)
    return explanation


def process_text():
    text_in = text_input.get("1.0", "end-1c")
    if not text_in.strip():
        result_label.config(text="Please enter some text.")
        return

    result_label.config(text="Processing, please wait...")
    root.update()

    # Perform classification
    sentiment = model.predict(vectorizer.transform([str(text_in)]))[0]

    if sentiment == "positive":
        result_label.config(text=f"The text is positive")
    else:
        result_label.config(text=f"The text is negative")

    explain_button.config(state=tk.NORMAL)


def explain_text():
    text_in = text_input.get("1.0", "end-1c")
    if not text_in.strip():
        result_label.config(text="Please enter some text.")
        return

    explain_button.config(state=tk.DISABLED)
    result_label.config(text="Generating explanation, please wait...")
    threading.Thread(target=_explain_text, args=(text_in,)).start()


def _explain_text(text_in):
    explanation = explain_instance(text_in)

    # Clear previous explanation rows
    result_label.config(text="Explanation:")

    # Display each feature and its weight in a new row
    for feature, weight in explanation.as_list():
        current_text = result_label.cget("text")
        result_label.config(text=current_text + f"\n{feature}: {weight}")

    explain_button.config(state=tk.NORMAL)


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

    explain_button = tk.Button(root, text="EXPLAIN", command=explain_text, state=tk.DISABLED)
    explain_button.grid(row=0, column=2, padx=10, pady=10)

    result_label = tk.Label(root, text="", wraplength=400)
    result_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

    clear_button = tk.Button(root, text="Clear", command=clear_text, fg='red')
    clear_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

    # Run the main loop
    root.mainloop()
