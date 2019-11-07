from record_module import *
from username_mfcc import *
from GUI.db import *

fname = "testfile.wav"

class Testing_file:
    def __init__(self):
        root = Toplevel()
        root.title("Speaker Recognition(Test)")
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.geometry("%dx%d" % (width, height))
        root.state('zoomed')

        ## Resizable Image

        image = Image.open('bggif/1.gif')
        global copy_of_image
        copy_of_image = image.copy()
        photo = ImageTk.PhotoImage(image)
        global label
        label = Label(root, image=photo)
        label.place(x=0, y=0, relwidth=1, relheight=1)
        label.bind('<Configure>', self.resize_image)

        ## Adding Buttons

        recording_button = Button(root, text="Record", bd=0, bg="black", fg="green", font=("Courier",35),command=record_audio)
        recording_button.place(relx=0.5, rely=0.35, anchor=CENTER)

        play_button = Button(root, text="Play", bd=0, bg="black", fg="green", font=("Courier",35),command=self.audioplay)
        play_button.place(relx=0.5, rely=0.5, anchor=CENTER)

        test1_button = Button(root, text="Test for Speaker", bd=0, bg="black", fg="green", font=("Courier",35),command = self.testaudio1)
        test1_button.place(relx=0.5, rely=0.65, anchor=CENTER)

        test2_button = Button(root, text="Test for Voxfage", bd=0, bg="black", fg="green", font=("Courier",35),command = self.testaudio2)
        test2_button.place(relx=0.5, rely=0.80, anchor=CENTER)

        root.mainloop()


    ## Function for resizing the Image

    def resize_image(self,event):
        new_width = event.width
        new_height = event.height
        global copy_of_image
        image = copy_of_image.resize((new_width, new_height))
        global photo
        photo = ImageTk.PhotoImage(image)
        global label
        label.config(image = photo)
        label.image = photo

    def audioplay(self):
        global fname
        #播放最后一次的音频
        play_audio(fname)

    def testaudio1(self):
    	k = test1(take=1)
    	recog(k)

    def testaudio2(self):
    	k = test1(take=0)
    	recog(k)

