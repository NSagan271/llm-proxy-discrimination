import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from llm_attr_inf.dataset.base import Dataset

def edit_text_snippets(snippets, highest_seen: int = -1):
    """
    Open a GUI to edit text snippets one-by-one.
    Returns the edited list after the window is closed.
    """
    original = snippets.copy()
    edited = snippets.copy()
    idx = 0

    root = tk.Tk()
    root.title("Redact Personal Information")
    root.geometry("800x600")

    # ---------------- Reset logic ----------------
    def reset_current():
        text.delete("1.0", tk.END)
        text.insert(tk.END, original[idx])
        edited[idx] = original[idx]

    # ---------------- Header ----------------
    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", pady=5, padx=10)

    header_frame.columnconfigure(0, weight=1)
    header_frame.columnconfigure(1, weight=0)

    header = ttk.Label(
        header_frame,
        text="",
        font=tkfont.Font(family="consolas", size=14)
    )
    header.grid(row=0, column=0, sticky="n")

    ttk.Button(
        header_frame,
        text="Reset",
        command=reset_current
    ).grid(row=0, column=1, padx=10)

    # ---------------- Navigation / Jump bar ----------------
    nav_frame = ttk.Frame(root)
    nav_frame.pack(fill="x", padx=10, pady=(0, 5))

    ttk.Label(nav_frame, text="Jump to:", font=tkfont.Font(family="consolas", size=12)).pack(side="left")

    jump_var = tk.StringVar()
    jump_entry = ttk.Entry(nav_frame, width=6, textvariable=jump_var, font=tkfont.Font(family="consolas", size=12))
    jump_entry.pack(side="left", padx=5)

    highest_seen_label = ttk.Label(nav_frame, text="", font=tkfont.Font(family="consolas", size=12))
    highest_seen_label.pack(side="right")

    def jump_to():
        nonlocal idx, highest_seen
        try:
            target = int(jump_var.get()) - 1  # user-facing is 1-indexed
        except ValueError:
            return

        if 0 <= target < len(edited):
            save_current()
            highest_seen = max(highest_seen, idx, target)
            idx = target
            update_view()

    ttk.Button(nav_frame, text="Go", command=jump_to).pack(side="left")

    # ---------------- Text box ----------------
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
        font=text_font,
        insertbackground="black"
    )
    text.pack(fill="both", expand=True)
    scrollbar.config(command=text.yview)

    # ---------------- Buttons ----------------
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=8)

    def save_current():
        edited[idx] = text.get("1.0", tk.END).rstrip()

    def update_view():
        seen_text = " (seen)" if idx <= highest_seen else ""
        header.config(text=f"Snippet {idx + 1} / {len(edited)}{seen_text}")
        highest_seen_label.config(
            text=f"Highest seen: {highest_seen + 1 if highest_seen >= 0 else '—'}"
        )
        text.delete("1.0", tk.END)
        text.insert(tk.END, edited[idx])

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
