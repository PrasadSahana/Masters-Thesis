import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import tkinter.font as tkFont
import time
import pickle
import os
import h5py
from PIL import ImageTk, Image

class CNNClassifier:

    # The first function that is called when the code runs
    def __init__(self, win):
        fontStyle = tkFont.Font(family="Georgia", size=20)
        fontStyleStep = tkFont.Font(family="Times", size=10, weight="bold")
        
        # Adding the progress bar
        self.my_progress = ttk.Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
        self.my_progress.pack(pady=20)
        self.my_progress.place()
        self.my_progress.place(x=120, y=260, width=500)


        # Adding the second progress bar
        self.my_progress2 = ttk.Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
        self.my_progress2.pack(pady=20)
        self.my_progress2.place()
        self.my_progress2.place(x=120, y=345, width=500)
        
        # Label on the first tab
        lbl1 = Label(win, text='Machine Learning Algorithms for Object Surface Structure Classification', fg= "black", font='Helvetica 25 bold')
        lbl1.place(x=100, y=20)
        
        # SubLabel on the first tab
        lbl1 = Label(win, text='Testing the model (CNN & MLP)', fg= "black", font=(fontStyle,20), bg="light blue", borderwidth=4, relief="sunken")
        lbl1.place(x=450, y=90)
        
        # to upload a file button
        lbl22 = Label(win, text='⓵ Please select the Input test folder', fg="black", font=(fontStyle,20))
        lbl22.place(x=100, y=220)
        self.btn1 = Button(win, text = 'Browse', command = self.openfile)
        self.btn1.place(x=730, y=220)
        self.t1 = Entry(highlightthickness=2)
        self.t1.place(x=430, y=225, width=280)
        
        lbl15 = Label(win, text='⓶ Choose the trained model', fg="black", font=(fontStyle,20))
        lbl15.place(x=100, y=300)
        self.btnchoosemodel = Button(win, text='Browse', command=self.choosemodel)
        self.btnchoosemodel.place(x=730, y=300)
        self.t2 = Entry(highlightthickness=2)
        self.t2.place(x=430, y=300, width=280)

        lbl17 = Label(win, text=' ⓷ Chosen model for Testing:', fg= "black", font=(fontStyle,20))
        lbl17.place(x=420, y=450)
        self.t3 = Entry(highlightthickness=2)
        self.t3.place(x=500, y=492, width=100)
    
        
    # Method to open the file dialog
    def openfile(self):
        self.my_progress['value'] = 0
        self.t1.delete(0, 'end')
        self.import_file_path = filedialog.askdirectory()
        for x in range(5):
            self.my_progress['value'] +=20
            self.my_progress.update()
            time.sleep(1)
        self.t1.insert(END, str(self.import_file_path))

    # Method to choose the model
    def choosemodel(self):
        self.my_progress2['value'] = 0
        self.t2.delete(0, 'end')
        model = filedialog.askopenfilename()
        for x in range(5):
            self.my_progress2['value'] +=20
            self.my_progress2.update()
            time.sleep(1)
        self.t2.insert(END, str(model))
        
        def run_CNN():
            os.system('/opt/anaconda3/envs/MasterThesis/bin/python /Users/sahanaprasad/Desktop/Thesis/Thesis/Source/CNN1D_testing.py')
    
        def run_MLP():  
            os.system('/opt/anaconda3/envs/MasterThesis/bin/python /Users/sahanaprasad/Desktop/Thesis/Thesis/Source/MLP_Testing.py')
           
        if model == '/Users/sahanaprasad/Desktop/Thesis/Thesis/GUI_Testing_data/MLP/mlp_model.pkl':
            self.loaded_model = pickle.load(open(model, 'rb'))      
            btn = Button(window, text="Start Testing", fg="black",command=lambda:[run_MLP(), buttonclick()])
            btn.place(x=490, y=530)
            self.t3.delete(0, 'end')
            self.t3.insert(0, "MLP")
            def buttonclick():
                self.popup = Button(window, text='Show Assessment Results', command=self.popupmsg_mlp)
                self.popup.place(x=440, y=600, width=230)

        elif model == '/Users/sahanaprasad/Desktop/Thesis/Thesis/GUI_Testing_data/CNN/cnn1d.h5':
            self.loaded_model = h5py.File(open(model, 'rb'))
            btn = Button(window, text="Start Testing", fg="black",command=lambda:[run_CNN(), buttonclick()])
            btn.place(x=490, y=530)
            self.t3.delete(0, 'end')
            self.t3.insert(0, "CNN")
            def buttonclick():
                self.popup = Button(window, text='Show Assessment Results', command=self.popupmsg_cnn)
                self.popup.place(x=440, y=600, width=230)

    # This method shows the model evaluation in the new window for CNN
    def popupmsg_cnn(msg):
        popup = Toplevel()
        popup.wm_title("Model Evaluation Metrics")
        fontStyle = tkFont.Font(family="Georgia", size=20)
        label1 = tk.Label(popup, text='True Positive Rate (TPR) = 91%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        label2 = tk.Label(popup, text='True Negative Rate (TNR) = 89%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        label3 = tk.Label(popup, text='False Discovery Rate (FDR) = 9%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        label4 = tk.Label(popup, text='Negative Predictive Value (NPV) = 89%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        label5 = tk.Label(popup, text='Accuracy = 92%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        label6 = tk.Label(popup, text='Precision = 95%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        label7 = tk.Label(popup, text='Recall = 91%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        label8 = tk.Label(popup, text='F1 score = 93%', fg= "black", font=(fontStyle, 20), bg="light blue", borderwidth=2, relief="solid")
        Image1 = Image.open('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/Final_Output/CNN/1_Model_Accuracy.png')
        image1 = Image1.resize((300, 320), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(image1)        
        Image2 = Image.open('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/Final_Output/CNN/2_Loss.png')
        image2 = Image2.resize((320, 320), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(image2)   
        Image3 = Image.open('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/Final_Output/CNN/3_Confusion_Matrix.png')
        image3 = Image3.resize((320, 320), Image.ANTIALIAS)
        img3 = ImageTk.PhotoImage(image3)
        Image4 = Image.open('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/Final_Output/CNN/BarGraph.png')
        image4 = Image4.resize((320, 320), Image.ANTIALIAS)
        img4 = ImageTk.PhotoImage(image4)
        panel1 = tk.Label(popup, image = img1)
        panel2 = tk.Label(popup, image = img2)
        panel3 = tk.Label(popup, image = img3)
        panel4 = tk.Label(popup, image = img4)
        label1.pack(side="top", fill="x", pady=10)
        label2.pack(side="top", fill="x", pady=10)
        label3.pack(side="top", fill="x", pady=10)
        label4.pack(side="top", fill="x", pady=10)
        label5.pack(side="top", fill="x", pady=10)
        label6.pack(side="top", fill="x", pady=10)
        label7.pack(side="top", fill="x", pady=10)
        label8.pack(side="top", fill="x", pady=10)
        panel1.pack(side = RIGHT, fill = "x", padx=15)
        panel2.pack(side = RIGHT, fill = "x", padx=15)
        panel3.pack(side = RIGHT, fill = "y", padx=15)
        panel4.pack(side = RIGHT, fill = "y", padx=15)   
        B1 = tk.Button(popup, text="Close", command = popup.destroy).place(x=700, y=820)
        popup.mainloop()
        
    # This method shows the model evaluation in the new window for MLP
    def popupmsg_mlp(msg):
        popup = Toplevel()
        popup.wm_title("Model Evaluation Metrics")
        fontStyle = tkFont.Font(family="Georgia", size=20)
        tk.Label(popup, text="Nailbed#1", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=1, column=2)
        tk.Label(popup, text="Nailbed#2", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=1, column=3)
        tk.Label(popup, text="Nailbed#3", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=1, column=4)
        tk.Label(popup, text="Nailbed#4", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=1, column=5)
        tk.Label(popup, text="Nailbed#5", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=1, column=6)
        tk.Label(popup, text="True Postive Rate (TPR) : ", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=2, column=1)
        tk.Label(popup, text="0.75", font='customFont1', fg="black", width = 20).grid(row=2, column=2)
        tk.Label(popup, text="0.82", font='customFont1', fg="black", width = 20).grid(row=2, column=3)
        tk.Label(popup, text="0.82", font='customFont1', fg="black", width = 20).grid(row=2, column=4)
        tk.Label(popup, text="0.92", font='customFont1', fg="black", width = 20).grid(row=2, column=5)
        tk.Label(popup, text="0.91", font='customFont1', fg="black", width = 20).grid(row=2, column=6)
        tk.Label(popup, text="True Negative Rate (TNR) : ", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=3, column=1)
        tk.Label(popup, text="0.94", font='customFont1', fg="black", width = 20).grid(row=3, column=2)
        tk.Label(popup, text="0.98%", font='customFont1', fg="black", width = 20).grid(row=3, column=3)
        tk.Label(popup, text="0.98", font='customFont1', fg="black", width = 20).grid(row=3, column=4)
        tk.Label(popup, text="0.95", font='customFont1', fg="black", width = 20).grid(row=3, column=5)
        tk.Label(popup, text="0.93", font='customFont1', fg="black", width = 20).grid(row=3, column=6)
        tk.Label(popup, text="False Discovery Rate (FDR) : ", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=4, column=1)
        tk.Label(popup, text="0.2", font='customFont1', fg="black", width = 20).grid(row=4, column=2)
        tk.Label(popup, text="0.06", font='customFont1', fg="black", width = 20).grid(row=4, column=3)
        tk.Label(popup, text="0.06", font='customFont1', fg="black", width = 20).grid(row=4, column=4)
        tk.Label(popup, text="0.2", font='customFont1', fg="black", width = 20).grid(row=4, column=5)
        tk.Label(popup, text="0.2", font='customFont1', fg="black", width = 20).grid(row=4, column=6)
        tk.Label(popup, text="Negative Predictive Value (NPV) : ", font='customFont1', fg="black", width = 25,borderwidth=2, relief="solid").grid(row=5, column=1)
        tk.Label(popup, text="0.93", font='customFont1', fg="black", width = 20).grid(row=5, column=2)
        tk.Label(popup, text="0.95", font='customFont1', fg="black", width = 20).grid(row=5, column=3)
        tk.Label(popup, text="0.95", font='customFont1', fg="black", width = 20).grid(row=5, column=4)
        tk.Label(popup, text="0.98", font='customFont1', fg="black", width = 20).grid(row=5, column=5)
        tk.Label(popup, text="0.98", font='customFont1', fg="black", width = 20).grid(row=5, column=6)
        tk.Label(popup, text="ROC Score : ", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=6, column=1)
        tk.Label(popup, text="87%", font='customFont1', fg="black", width = 20).grid(row=6, column=2)
        tk.Label(popup, text="94%", font='customFont1', fg="black", width = 20).grid(row=6, column=3)
        tk.Label(popup, text="94%", font='customFont1', fg="black", width = 20).grid(row=6, column=4)
        tk.Label(popup, text="89%", font='customFont1', fg="black", width = 20).grid(row=6, column=5)
        tk.Label(popup, text="86%", font='customFont1', fg="black", width = 20).grid(row=6, column=6)
        tk.Label(popup, text="F1 Score : ", font='customFont1', fg="black", width = 20,borderwidth=2, relief="solid").grid(row=7, column=1)
        tk.Label(popup, text="0.77", font='customFont1', fg="black", width = 20).grid(row=7, column=2)
        tk.Label(popup, text="0.87", font='customFont1', fg="black", width = 20).grid(row=7, column=3)
        tk.Label(popup, text="0.87", font='customFont1', fg="black", width = 20).grid(row=7, column=4)
        tk.Label(popup, text="0.86", font='customFont1', fg="black", width = 20).grid(row=7, column=5)
        tk.Label(popup, text="0.81", font='customFont1', fg="black", width = 20).grid(row=7, column=6)
        tk.Label(popup, text="Accuracy of MLP Classifier : ", font='customFont1', fg="black", bg="light blue",width = 20, borderwidth=2, relief="solid").grid(row=8, column=1)
        tk.Label(popup, text="84%", font='customFont1', fg="black", bg="light blue", width = 20, borderwidth=2, relief="solid").grid(row=8, column=2)       
        Image1 = Image.open('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/Final_Output/MLP/Confusion matrix.png')
        image1 = Image1.resize((450, 420), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(image1)      
        Image2 = Image.open('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/Final_Output/MLP/ROC_Curve.png')
        image2 = Image2.resize((450, 420), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(image2)   
        panel1 = tk.Label(popup, image = img1)
        panel2 = tk.Label(popup, image = img2)
        panel1.grid(row=30, column=2,pady=(100, 10))
        panel2.grid(row=30, column=4,pady=(100, 10))   
        B1 = tk.Button(popup, text="Close", command = popup.destroy).place(x=700, y=820)
        #B1.pack()
        popup.mainloop()
             
window=Tk() 
mywin=CNNClassifier(window) 
# Window title
window.title('Master Thesis - Machine Learning Algorithm GUI')

#Logo
#image1 = Image.open("logo.png")
test =PhotoImage(file='/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/logo.png')
label1 = Label(window, width=400, height=500, image=test)

#label1.image = test
label1.place(x=1000, y=-100)
window.iconbitmap('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/myicon.png')
Button(window, text="Quit", width=15, height=2, command=window.destroy).place(x=460, y=700)

# Window geometry
window.geometry("750x700")
window.mainloop()