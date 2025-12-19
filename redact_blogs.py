import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from llm_attr_inf.dataset.base import Dataset

def edit_text_snippets(snippets, highest_seen: int=-1):
    """
    Open a GUI to edit text snippets one-by-one.
    Returns the edited list after the window is closed.
    """
    original = snippets.copy()  # keep original for reset
    edited = snippets.copy()
    idx = 0

    root = tk.Tk()
    root.title("Redact Personal Information")

    root.geometry("800x500")


    # Reset button next to header
    def reset_current():
        """Reset only the current snippet to original"""
        text.delete("1.0", tk.END)
        text.insert(tk.END, original[idx])
        edited[idx] = original[idx]
    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", pady=5, padx=10)

    header_frame.columnconfigure(0, weight=1)
    header_frame.columnconfigure(1, weight=0)

    # Header label in column 0 (centered)
    header = ttk.Label(header_frame, text="", font=tkfont.Font(family="consolas", size=14))
    header.grid(row=0, column=0, sticky="n", padx=0)

    # Reset button in column 1
    ttk.Button(header_frame, text="Reset", command=reset_current).grid(row=0, column=1, padx=10)

    # Text box with scrollbar
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=5)

    scrollbar = ttk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    text_font = tkfont.Font(family="consolas", size=13)
    text = tk.Text(
        frame,
        wrap="word",
        yscrollcommand=scrollbar.set,
        fg="black",
        bg="white",
        height=5,
        font=text_font,
        insertbackground="black"
    )
    text.pack(fill="both", expand=True)
    scrollbar.config(command=text.yview)

    # Button row
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=8)

    def update_view():
        seen_text = " (seen)" if idx <= highest_seen else ""
        header.config(text=f"Snippet {idx + 1} / {len(edited)}{seen_text}")
        text.delete("1.0", tk.END)
        text.insert(tk.END, edited[idx])

    def save_current():
        edited[idx] = text.get("1.0", tk.END).rstrip()

    def next_item(event=None):
        nonlocal idx, highest_seen
        save_current()
        if idx < len(edited) - 1:
            highest_seen = max(highest_seen, idx)
            idx += 1
            update_view()

    def prev_item():
        nonlocal idx
        save_current()
        if idx > 0:
            idx -= 1
            update_view()

    def finish():
        save_current()
        root.destroy()

    ttk.Button(btn_frame, text="Previous", command=prev_item).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Next", command=next_item).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="Finish", command=finish).pack(side="left", padx=5)

    root.bind("<Control-Return>", next_item)

    update_view()
    root.mainloop()

    return edited, highest_seen


if __name__ == "__main__":
    REDACTED_PATH = "outputs/data/blogs_redacted"
    if os.path.exists(REDACTED_PATH):
        dataset = Dataset.load(REDACTED_PATH)
        with open(f"{REDACTED_PATH}/highest_seen.txt", "r") as f:
            highest_seen = int(f.readline())
    else:
        dataset = Dataset.load("outputs/data/blogs")
        highest_seen = -1
    
    texts = sum([[t.text for t in texts] for texts in dataset.texts], start=[])
    out, highest_seen = edit_text_snippets(texts[:], highest_seen=highest_seen)

    idx = 0
    for tt in dataset.texts:
        for t in tt:
            t.text = out[idx]
            idx += 1
    assert idx == len(texts)
    dataset.save(REDACTED_PATH)
    with open(f"{REDACTED_PATH}/highest_seen.txt", "w") as f:
        f.write(str(highest_seen))
