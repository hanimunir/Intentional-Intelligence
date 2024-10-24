from Tkinter import*

class GUIDemo(Frame): # (inherit) Tkinter Frame

    def	__init__(self, master=None):

        Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        # input
        self.inputText = Label(self)
        self.inputText["text"] = "Input:"
        self.inputText.grid(row=0, column=0)
        self.inputField = Entry(self)
        self.inputField["width"] = 50
        self.inputField.grid(row=0, column=1, columnspan=6)
        #output
        self.outputText = Label(self)
        self.outputText["text"] = "Output:"
        self.outputText.grid(row=1, column=0)
        self.outputField = Entry(self)
        self.outputField["width"] = 50
        self.outputField.grid(row=1, column=1, columnspan=6)
        self.new = Button(self)  
        self.new["text"] = "New"  
        self.new.grid(row=2, column=0)
        self.load = Button(self)
	self.load = Button(self)
	self.load["text"] = "Load"
	self.load.grid(row=2, column=1)
	self.encode = Button(self)
	self.encode["text"] = "Encode"
	self.encode.grid(row=2, column=3)

	self.decode = Button(self)
	self.decode["text"] = "Decode"
	self.decode.grid(row=2, column=4)
	self.clear = Button(self)
	self.clear["text"] = "Clear"
	self.clear.grid(row=2, column=5)

	self.copy = Button(self)
	self.copy["text"] = "Copy"
	self.copy.grid(row=2, column=6)

	self.displayText = Label(self)
	self.displayText["text"] = "something happened"
	self.displayText.grid(row=3, column=0, columnspan=7



if __name__=='__main__':
	root = Tk()
	app = GUIDemo(master=root)
	app.mainloop()
