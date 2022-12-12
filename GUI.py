import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import toolkit as ttk

import tkinter as tk


#fig, ax = plt.subplots(2,4)
class MyApp(tk.Tk):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.configure(background='white')

		self.Train_Path = tk.StringVar()
		self.Test_Path = tk.StringVar()

		self.title('hello world')
		self.geometry('720x720')
		self.setLabel("Train_Path", (20, 500))
		self.setLabel("Test_Path", (20, 540))

		self.setEntry(self.Train_Path, (200, 500), 'Bonus_Training.txt', 30)
		self.setEntry(self.Test_Path, (200, 540), 'Bonus_Testing.txt', 30)


		self.setButton("Train!", (20, 600), self.Train)

	def setLabel(self, text, pos):
		label = tk.Label(self, text=text, font=("MV Boli", 16), bg="white")
		label.place(x=pos[0], y=pos[1])

	def setEntry(self, re, pos, default, width):
		entry = tk.Entry(self, font=("MV Boli", 16), textvariable=re, width=width)
		entry.insert(-1, default)
		entry.place(x=pos[0], y=pos[1])

	def setButton(self, text, pos, fun):
		button = tk.Button(self, text=text, font=("MV Boli", 16), command=fun)
		button.place(x=pos[0], y=pos[1])

	def Train(self):
		train = ttk.loadData(str(self.Train_Path.get()))
		test = ttk.loadData(str(self.Test_Path.get()))
		c, h, w = train.shape; n = h*w

		hopfield = ttk.HopfieldNetwork(n)
		hopfield.train(train)

		self.fig, self.ax = plt.subplots(2,len(train))
		for i in range(len(test)):
			result = hopfield.recall(test[i].flatten())
			result = result.reshape(h, w)
			self.ax[0][0].set_ylabel('Train')
			self.ax[0][i].matshow(train[i])
			self.ax[1][0].set_ylabel('Test')
			self.ax[1][i].matshow(result)		

		
		self.canvs = FigureCanvasTkAgg(self.fig, self)
		self.canvs.draw()
		self.canvs.get_tk_widget().pack()

app = MyApp()
app.mainloop()