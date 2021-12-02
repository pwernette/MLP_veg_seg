import sys
import tkinter as tk
from tkinter import *

# # create window instance
# win = Tk()
# win.title('Specify Model Name')

# def getinput(textboxvar):
#     global modelname
#     modelname = textboxvar.get("1.0",'end-1c')
#     win.destroy()

# # create entry widget to accept user input
# textbox = Text(win, height=2, width=15)
# textbox.pack()
#
# # create validation button
# buttonconfirm = Button(win, text='Confirm', width=20, command=getinput)
# buttonconfirm.pack()
#
# win.mainloop()
# print(modelname)

def getmodelname():
    # create window instance
    win = Tk()
    win.title('Specify Model Name')

    # create entry widget to accept user input
    textbox = Text(win, height=1, width=50)
    textbox.pack()

    def getinput():
        global modelname
        modelname = textbox.get('1.0','end-1c').split('\n')[0]
        win.destroy()
        return modelname

    def cancel_and_exit():
        win.destroy()
        sys.exit('No model name specified. Exiting program.')

    # create validation button
    buttonconfirm = Button(win,
                            text='Confirm Model Name',
                            width=40,
                            command=lambda:getinput())
    buttonconfirm.pack(pady=5)

    win.bind('<Return>', lambda event:getinput())

    win.bind('<Escape>', lambda event:cancel_and_exit())

    win.mainloop()
    if ' ' in modelname:
        mname = modelname.replace(' ','_')
    else:
        mname = modelname
    return(mname)

foo = getmodelname()
print(foo)
print(foo)
