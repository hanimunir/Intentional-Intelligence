from Tkinter import*
import mysql.connector

class GUIDemo(Frame): # (inherit) Tkinter Frame

    def	__init__(self, master=None):

        Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def buttonPushed(self,inputvalue):

		mydb = mysql.connector.connect(
  		host="localhost",
  		user="yourusername",
  		password="yourpassword",
  		database="mydatabase"
		)


		mycursor = mydb.cursor()

		sql = "INSERT INTO your_database_name(input) VALUES (%s)"
		val = (inputvalue)
		mycursor.execute(sql, val)

		mydb.commit()



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
	self.save = Button(self,command=buttonPushed(self.inputText["text"]) )  
	self.save["text"] = "Save"  
	self.save.grid(row=2, column=2)


if __name__=='__main__':
	root = Tk()
	app = GUIDemo(master=root)
	app.mainloop()
