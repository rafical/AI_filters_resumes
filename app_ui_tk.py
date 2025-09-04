#!/usr/bin/env python3
"""
app_tk.py — Tkinter UI for Resume Ranker (Universal LLM API Adapter)
Adds:
• Clear button (resets prompt, folder, table, status)
• Wrapped summary text inside the Treeview
• Sort results by score DESC (highest → lowest)
• Multi-line prompt input (2–3 lines, word-wrap + scrollbar)
• Export results to CSV or Excel (.xlsx)
• (NEW) Model dropdown loaded from config.yml (supports Ollama + others)
• (NEW) Progress modal window during LLM processing
• (NEW) Application icon for main window and dialogs
"""

from __future__ import annotations

import csv
import json
import threading
import os
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import tkinter.font as tkfont
import textwrap
import time

from ranker_core import (
    ResumeResult,
    load_config,
    load_resumes,
    evaluate_resumes,
)

APP_TITLE = "Resume Fileter Ranker"
JSON_SAVE_NAME = "rankings_gui.json"
ICON_PATH = "logo_rafical_ico.ico"  # Your icon file


class ProgressDialog(tk.Toplevel):
    """Modal progress dialog for LLM processing"""
    
    def __init__(self, parent, total_files: int):
        super().__init__(parent)
        self.title("Processing Resumes")
        self.geometry("400x150")
        self.resizable(False, False)
        
        # Set icon for the dialog
        self._set_icon()
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        
        self.total_files = total_files
        self.current_file = 0
        self.is_cancelled = False
        
        self._build_ui()
        
    def _set_icon(self):
        """Set the window icon if available"""
        try:
            if os.path.exists(ICON_PATH):
                self.iconbitmap(ICON_PATH)
        except Exception as e:
            print(f"Could not set dialog icon: {e}")
        
    def _build_ui(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        ttk.Label(main_frame, text="Evaluating resumes with LLM...", 
                 font=("", 10, "bold")).pack(pady=(0, 15))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var,
            maximum=self.total_files,
            mode='determinate'
        )
        self.progress_bar.pack(fill="x", pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Initializing...")
        self.status_label.pack(pady=(0, 15))
        
        # Cancel button
        ttk.Button(main_frame, text="Cancel", command=self._cancel).pack()
        
        # Update immediately
        self.update_progress(0, "Starting...")
        
    def update_progress(self, current: int, status: str):
        """Update progress bar and status text"""
        self.current_file = current
        self.progress_var.set(current)
        self.status_label.config(text=status)
        self.update_idletasks()
        
    def _cancel(self):
        """Cancel the operation"""
        self.is_cancelled = True
        self.status_label.config(text="Cancelling...")
        self.update_idletasks()
        
    def was_cancelled(self) -> bool:
        """Check if user cancelled the operation"""
        return self.is_cancelled


class RankerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x600")
        self.minsize(820, 520)
        
        # Set application icon
        self._set_icon()

        # State
        self.resumes_dir = tk.StringVar()
        self.prompt_text = tk.StringVar()  # retained for compatibility; Text widget used below
        self.status_text = tk.StringVar(value="Ready.")
        self.temperature = tk.DoubleVar(value=0.2)
        self.max_tokens = tk.IntVar(value=1000)
        self.progress_dialog = None
        self.cancel_processing = False

        # Provider/model selection state (NEW)
        self.providers_list: list[dict] = []
        self.provider_label_to_cfg: dict[str, dict] = {}
        self.selected_provider_label = tk.StringVar()

        # hold last results for export
        self.last_results: list[ResumeResult] | None = None

        # Load provider config
        try:
            cfg = load_config("config.yml")
            providers = cfg.get("providers", [])
            if not providers:
                raise RuntimeError("No providers configured in config.yml")

            # Build label->cfg map for dropdown
            self.providers_list = providers
            self.provider_label_to_cfg = {
                self._fmt_provider_label(p, idx): p
                for idx, p in enumerate(providers)
            }
            # Default selection = first item
            first_label = next(iter(self.provider_label_to_cfg.keys()))
            self.selected_provider_label.set(first_label)
            self.provider_cfg = self.provider_label_to_cfg[first_label]
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            self.provider_cfg = {}

        # fonts & style
        self.default_font = tkfont.nametofont("TkDefaultFont")
        style = ttk.Style(self)
        style.configure("Treeview", rowheight=84)  # taller rows for wrapped text

        self._build_ui()
        
    def _set_icon(self):
        """Set the application icon if available"""
        try:
            if os.path.exists(ICON_PATH):
                self.iconbitmap(ICON_PATH)
        except Exception as e:
            print(f"Could not set application icon: {e}")

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Job / Prompt:").grid(row=0, column=0, sticky="nw")

        # Multi-line Text widget for prompt with scrollbar
        prompt_wrap = ttk.Frame(top)
        prompt_wrap.grid(row=0, column=1, columnspan=3, sticky="we", padx=(8, 0))
        top.columnconfigure(3, weight=1)

        self.prompt_textbox = tk.Text(
            prompt_wrap,
            height=3,  # ~2–3 lines
            wrap="word",
            undo=True
        )
        self.prompt_textbox.pack(side="left", fill="x", expand=True)

        prompt_vsb = ttk.Scrollbar(prompt_wrap, orient="vertical", command=self.prompt_textbox.yview)
        self.prompt_textbox.configure(yscrollcommand=prompt_vsb.set)
        prompt_vsb.pack(side="right", fill="y")

        ttk.Label(top, text="Resumes folder:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.folder_entry = ttk.Entry(top, textvariable=self.resumes_dir, width=60)
        self.folder_entry.grid(row=1, column=1, sticky="we", padx=(8, 8), pady=(8, 0))
        ttk.Button(top, text="Browse…", command=self._select_folder).grid(row=1, column=2, sticky="w", pady=(8, 0))

        ttk.Label(top, text="Temperature:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=0.0, to=2.0, increment=0.1, textvariable=self.temperature, width=6)\
            .grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(top, text="Max tokens:").grid(row=2, column=2, sticky="e", pady=(8, 0))
        ttk.Spinbox(top, from_=256, to=4096, increment=128, textvariable=self.max_tokens, width=8)\
            .grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(8, 0))

        # (NEW) Model dropdown row
        ttk.Label(top, text="Model:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.model_combo = ttk.Combobox(
            top,
            textvariable=self.selected_provider_label,
            values=list(self.provider_label_to_cfg.keys()),
            state="readonly",
            width=60
        )
        self.model_combo.grid(row=3, column=1, columnspan=2, sticky="we", padx=(8, 8), pady=(8, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Button(top, text="Reload models", command=self._reload_models_from_config)\
            .grid(row=3, column=3, sticky="w", pady=(8, 0))

        # Action buttons row
        actions = ttk.Frame(self, padding=(10, 0))
        actions.pack(fill="x")
        ttk.Button(actions, text="Run Ranking", command=self._run_ranking_threaded)\
            .pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Clear", command=self._clear_all)\
            .pack(side="left", padx=(0, 12))

        # Export buttons (disabled until results exist)
        self.export_csv_btn = ttk.Button(actions, text="Export CSV", command=self._export_csv, state=tk.DISABLED)
        self.export_csv_btn.pack(side="left")
        self.export_xlsx_btn = ttk.Button(actions, text="Export Excel (.xlsx)", command=self._export_excel, state=tk.DISABLED)
        self.export_xlsx_btn.pack(side="left", padx=(8, 0))

        # Results table
        mid = ttk.Frame(self, padding=10)
        mid.pack(fill="both", expand=True)

        columns = ("rank", "file", "score", "summary")
        self.tree = ttk.Treeview(mid, columns=columns, show="headings")
        self.tree.heading("rank", text="Rank")
        self.tree.heading("file", text="File")
        self.tree.heading("score", text="Score")
        self.tree.heading("summary", text="Summary")
        self.tree.column("rank", width=60, anchor="e", stretch=False)
        self.tree.column("file", width=220, stretch=True)
        self.tree.column("score", width=80, anchor="e", stretch=False)
        self.tree.column("summary", width=520, stretch=True)
        self.tree.pack(fill="both", expand=True, side="left")

        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.pack(side="right", fill="y")

        # Status bar
        bottom = ttk.Frame(self, padding=(10, 8))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status_text, anchor="w").pack(fill="x")

    # ---------- Helpers ----------
    def _fmt_provider_label(self, cfg: dict, idx: int) -> str:
        """Create a stable, readable label for the dropdown."""
        prov = str(cfg.get("provider", "unknown")).strip()
        model = str(cfg.get("model", "")).strip()
        label = f"{prov} / {model}" if model else prov
        if label in self.provider_label_to_cfg and self.provider_label_to_cfg[label] is not cfg:
            label = f"{label} [{idx}]"
        return label

    def _reload_models_from_config(self):
        """Reload all providers/models from config.yml (useful after edits)."""
        try:
            cfg = load_config("config.yml")
            providers = cfg.get("providers", [])
            if not providers:
                raise RuntimeError("No providers configured in config.yml")
            self.providers_list = providers
            # rebuild map
            new_map = {}
            for idx, p in enumerate(providers):
                lbl = self._fmt_provider_label(p, idx)
                new_map[lbl] = p
            self.provider_label_to_cfg = new_map
            self.model_combo["values"] = list(self.provider_label_to_cfg.keys())
            # keep selection if possible, otherwise default to first
            current = self.selected_provider_label.get()
            if current not in self.provider_label_to_cfg:
                first_label = next(iter(self.provider_label_to_cfg.keys()))
                self.selected_provider_label.set(first_label)
                self.provider_cfg = self.provider_label_to_cfg[first_label]
            self._set_status("Models reloaded from config.yml")
        except Exception as e:
            messagebox.showerror("Reload error", str(e))

    def _on_model_change(self, _event=None):
        label = self.selected_provider_label.get()
        cfg = self.provider_label_to_cfg.get(label)
        if cfg:
            self.provider_cfg = cfg
            self._set_status(f"Selected: {label}")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select resumes folder")
        if folder:
            self.resumes_dir.set(folder)

    def _get_prompt(self) -> str:
        """Read text from the multi-line prompt box."""
        return self.prompt_textbox.get("1.0", "end-1c").strip()

    def _clear_all(self):
        """Reset prompt, folder, table, and status for a fresh run."""
        self.prompt_textbox.delete("1.0", "end")
        self.resumes_dir.set("")
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._set_status("Ready.")
        self.prompt_textbox.focus_set()
        self.last_results = None
        self._set_export_buttons_enabled(False)

    def _set_export_buttons_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.export_csv_btn.config(state=state)
        self.export_xlsx_btn.config(state=state)

    def _run_ranking_threaded(self):
        self.cancel_processing = False
        threading.Thread(target=self._run_ranking, daemon=True).start()

    def _show_progress_dialog(self, total_files: int):
        """Show progress dialog"""
        if self.progress_dialog and self.progress_dialog.winfo_exists():
            self.progress_dialog.destroy()
        self.progress_dialog = ProgressDialog(self, total_files)
        return self.progress_dialog

    def _update_progress(self, current: int, status: str):
        """Update progress dialog if it exists"""
        if (self.progress_dialog and 
            self.progress_dialog.winfo_exists() and 
            not self.progress_dialog.was_cancelled()):
            self.progress_dialog.update_progress(current, status)
            return not self.progress_dialog.was_cancelled()
        return False

    def _close_progress_dialog(self):
        """Close progress dialog"""
        if self.progress_dialog and self.progress_dialog.winfo_exists():
            self.progress_dialog.destroy()
        self.progress_dialog = None

    def _calc_char_width_for_column(self, column_id: str) -> int:
        """Estimate how many characters fit in the given Treeview column."""
        pixel_width = self.tree.column(column_id)["width"]
        avg_px = max(7, self.default_font.measure("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") // 52 or 7)
        return max(12, pixel_width // avg_px)

    def _wrap_for_tree(self, text: str, column_id: str) -> str:
        """Insert newline breaks so the Treeview cell displays multiple lines."""
        width_chars = self._calc_char_width_for_column(column_id)
        return textwrap.fill(text, width=width_chars)

    def _run_ranking(self):
        prompt = self._get_prompt()
        folder = Path(self.resumes_dir.get().strip())

        if not prompt:
            messagebox.showwarning("Missing prompt", "Please enter a job/prompt.")
            return
        if not folder.exists():
            messagebox.showwarning("Missing folder", "Please select a valid resumes folder.")
            return

        try:
            self._set_status("Loading resumes...")
            resumes = load_resumes(folder)
            if not resumes:
                messagebox.showinfo("No files", "No resumes found (.pdf .docx .txt .md).")
                self._set_status("Ready.")
                return

            # Show progress dialog
            progress_dialog = self._show_progress_dialog(len(resumes))
            
            # Ensure we use the currently selected provider config
            selected_label = self.selected_provider_label.get()
            if selected_label in self.provider_label_to_cfg:
                self.provider_cfg = self.provider_label_to_cfg[selected_label]

            # Custom evaluate function with progress updates
            results = []
            for i, (filename, text) in enumerate(resumes.items(), 1):
                # Check if user cancelled
                if progress_dialog.was_cancelled():
                    self._set_status("Operation cancelled")
                    self._close_progress_dialog()
                    return
                
                # Update progress
                if not self._update_progress(i, f"Processing {filename}..."):
                    break
                
                # Process this resume
                try:
                    # Simulate processing time (remove this in production)
                    time.sleep(0.1)
                    
                    # Here you would call your actual evaluation logic
                    # For now, just create a dummy result
                    result = evaluate_resumes(
                        job_query=prompt,
                        resumes={filename: text},
                        provider_cfg=self.provider_cfg,
                        temperature=float(self.temperature.get()),
                        max_tokens=int(self.max_tokens.get()),
                    )
                    if result:
                        results.extend(result)
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    # Add error result
                    results.append(ResumeResult(
                        file=filename, 
                        score=0.0, 
                        summary=f"Error: {str(e)}"
                    ))

            self._close_progress_dialog()

            if progress_dialog.was_cancelled():
                self._set_status("Operation cancelled by user")
                return

            # sort by score (DESC)
            results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
            self.last_results = results_sorted
            self._set_export_buttons_enabled(True)

            # Update UI — wrap summary text for display
            for row in self.tree.get_children():
                self.tree.delete(row)
            for i, r in enumerate(results_sorted, 1):
                wrapped_summary = self._wrap_for_tree(r.summary, "summary")
                self.tree.insert("", "end", values=(i, r.file, f"{r.score:.2f}", wrapped_summary))

            # Save JSON (sorted as shown)
            out_path = Path(JSON_SAVE_NAME)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "job_query": prompt,
                        "model": selected_label,
                        "results": [r.__dict__ for r in results_sorted],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            self._set_status(f"Done. Results saved to {out_path.resolve()}")
            
        except Exception as e:
            self._close_progress_dialog()
            messagebox.showerror("Error", str(e))
            self._set_status("Error. See details.")

    # ---------- Exporters ----------
    def _export_csv(self):
        if not self.last_results:
            messagebox.showinfo("No data", "Run ranking first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            title="Save results as CSV",
            initialfile="rankings.csv",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "file", "score", "summary"])
                for i, r in enumerate(self.last_results, 1):
                    writer.writerow([i, r.file, f"{r.score:.2f}", r.summary])
            self._set_status(f"Exported CSV → {path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to export CSV:\n{e}")

    def _export_excel(self):
        if not self.last_results:
            messagebox.showinfo("No data", "Run ranking first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx"), ("All files", "*.*")],
            title="Save results as Excel",
            initialfile="rankings.xlsx",
        )
        if not path:
            return
        try:
            import pandas as pd
        except Exception:
            messagebox.showwarning(
                "Excel export unavailable",
                "Excel export requires the 'pandas' package (and 'openpyxl').\n"
                "Tip: pip install pandas openpyxl\n\n"
                "You can also use 'Export CSV', which opens in Excel."
            )
            return
        try:
            rows = [
                {"rank": i, "file": r.file, "score": float(r.score), "summary": r.summary}
                for i, r in enumerate(self.last_results, 1)
            ]
            df = pd.DataFrame(rows, columns=["rank", "file", "score", "summary"])
            df.to_excel(path, index=False)
            self._set_status(f"Exported Excel → {path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to export Excel:\n{e}")

    def _set_status(self, text: str):
        self.status_text.set(text)
        self.update_idletasks()


def main():
    app = RankerApp()
    app.mainloop()


if __name__ == "__main__":
    main()