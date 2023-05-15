# LEGO sorter project
# Label selection dialog
# (c) lego-sorter team, 2022-2023

# This script implements TkInter label selection dialog which allows to choose
# an image label from list of known classes.
#
# Due to OpenCV's nature, the script has to be implemented separatelly,
# so it has to be called using subprocess() call. The --label flag
# might be used to provide it with currently selected label.
#
# If user choose a label from the list and clicks OK, 
# the label will be printed to stdout as {"label": <label>} JSON.

import os
import json
import tkinter as tk
from tkinter.simpledialog import Dialog
from string import ascii_lowercase, digits
from absl import app, flags

from lib.globals import IMAGE_DIR

FLAGS = flags.FLAGS
flags.DEFINE_string('label', None, help='Current label')

# copied from lib.image_dataset to avoid tensorflow load
def fast_get_class_names():
    return [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]

class SearchableListbox(tk.Listbox):
    """ Listbox with search and extra navigation """

    def __init__(self, master, content, value=None, **kwargs):
        self._content = content
        content_var = tk.StringVar()
        content_var.set(content)

        super().__init__(master, listvariable=content_var, **kwargs)

        if value:
            self._find(value)

        self.keybuf = ''
        self.bind('<KeyPress>', self._keypress)


    def cur_index(self):
        sel = self.curselection()
        return sel[0] if sel else -1

    def cur_value(self):
        sel = self.curselection()
        return self.get(sel) if sel else None

    def _find(self, val):
        for idx, line in enumerate(self._content):
            if line.startswith(val):
                self._select(idx)
                return idx
        return -1

    def _select(self, idx):
        self.select_clear(0, tk.END)
        self.select_set(idx)
        self.see(idx)
        self.activate(idx)
        self.selection_anchor(idx)

    def _clear(self):
        self.keybuf = ''

    def _keypress(self, event):
        key_lower = event.keysym.lower()
        match key_lower:
            case 'home':
                self.keybuf = ''
                self._select(0)

            case 'up':
                self.keybuf = ''
                idx = self.cur_index()
                if idx > 0:
                    self._select(idx-1)

            case 'down':
                self.keybuf = ''
                idx = self.cur_index()
                if idx < len(self._content)-1:
                    self._select(self.cur_index()+1)

            case 'end':
                self.keybuf = ''
                self._select(tk.END)

            case 'q':
                # this is to keep up with parent's way to close windows
                self.winfo_toplevel().destroy()

            case _:
                # save any valid symbol to keybuf and lookup
                if len(key_lower) == 1 and (key_lower in ascii_lowercase or key_lower in digits):
                    self.keybuf += event.keysym
                    self.after(1000, self._clear)
                    self._find(self.keybuf)

class SelectLabelDialog(Dialog):
    """ Label selection dialog """
    
    def body(self, master):
        cur_class = FLAGS.label
        class_list = fast_get_class_names()

        labelText = f'Current label is {cur_class}\n' if cur_class else ''
        labelText += 'Select new class label:'

        fillerPanel = tk.Frame(master)
        fillerPanel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(fillerPanel, anchor=tk.N, justify=tk.LEFT, text=labelText) \
            .pack(side=tk.TOP, fill=tk.BOTH, padx=10, expand=False)

        inputPanel = tk.Frame(fillerPanel)
        inputPanel.pack(side=tk.TOP, fill=tk.BOTH, ipadx=5, ipady=5, expand=True)

        self.classListBox = SearchableListbox(inputPanel, content=class_list, value=cur_class,
                    selectmode=tk.SINGLE, width=20, height=10)
        self.classListBox.pack(side=tk.LEFT, fill=tk.BOTH, ipadx=10, expand=True)

        classListBoxScroll = tk.Scrollbar(inputPanel, orient='vertical')
        classListBoxScroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.classListBox.config(yscrollcommand = classListBoxScroll.set)
        classListBoxScroll.config(command=self.classListBox.yview)

        # self.attributes("-toolwindow", True)
        self.focus_set()
        self.resizable(False, False)

        return self.classListBox

    def validate(self) -> bool:
        return self.classListBox.cur_value() is not None
        
    def apply(self):
        print(json.dumps({'label': self.classListBox.cur_value()}))

def main(_):
    root = tk.Tk('class_name')
    root.title('Label')

    # hide main window so only the dialog will be visible
    root.withdraw()
    SelectLabelDialog(root)

if __name__ == '__main__':
    app.run(main)
