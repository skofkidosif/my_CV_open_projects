import tkinter as tk
from PIL import Image
from PIL import ImageTk
import numpy as np
from tkinter import filedialog
from tkinter.ttk import Radiobutton
import cv2


class APP(tk.Frame):
    hv = 1
    acc = 1
    param1 = 50
    param2 = 30
    minRadius = 1
    maxRadius = 1

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
        self.hvalue1 = 0
        self.hvalue2 = 0
        self.hvalue3 = 0
        self.hvalue4 = 255
        self.hvalue5 = 255
        self.hvalue6 = 255
        self.lencnt1 = 5
        self.areacnt1 = 5
        self.areacnt2 = 7
        self.diamcnt = 7
        self.threshB = 127

    def initUI(self):
        self.parent.title("Sirius_CV")
        self.pack(fill=tk.BOTH, expand=1)

        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)

        self.label1 = tk.Label(self, border=15)
        self.label2 = tk.Label(self, border=15)
        self.tool_frame = tk.Frame(self, border=15)
        self.label1.grid(row=1, column=1)
        self.label2.grid(row=1, column=2)
        self.tool_frame.grid(row=1, column=0)

        fileMenu = tk.Menu(menubar)
        menubar.add_cascade(label="File", menu=fileMenu)

        fileMenu.add_command(label="Open", command=self.onOpen)
        fileMenu.add_command(label="Save", command=self.saveImage)

        basicMenu = tk.Menu(menubar)
        menubar.add_cascade(label="Basic", menu=basicMenu)

        circleMenu = tk.Menu(menubar)
        menubar.add_cascade(label="Circle", menu=circleMenu)

        contourMenu = tk.Menu(menubar)
        menubar.add_cascade(label="Contour", menu=contourMenu)

        basicMenu.add_command(label="Scale", command=self.onScale)
        basicMenu.add_command(label="Negative", command=self.onNeg)
        basicMenu.add_command(label="Origin", command=self.onPos)
        basicMenu.add_command(label="Gray", command=self.onGray)
        basicMenu.add_command(label="Threshold", command=self.on_threshold)
        basicMenu.add_command(label="ThresholdBin", command=self.on_thresh_bin)
        basicMenu.add_command(label="GaussianBlur", command=self.on_gauss)
        basicMenu.add_command(label="medianBlur", command=self.on_median)
        contourMenu.add_command(label="Canny", command=self.onCanny)
        basicMenu.add_command(label="CLAHE", command=self.onClahe)
        circleMenu.add_command(label="Hough", command=self.onHough)
        contourMenu.add_command(label="Contours", command=self.onCnt)

    def setImage(self):
        self.img = Image.open(self.fn)
        self.I = np.asarray(self.img)
        self.frame = cv2.imread(self.fn)
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        l, h = self.img.size
        text = str(2 * l + 100) + "x" + str(h + 50) + "+0+0"
        self.parent.geometry(text)
        photo = ImageTk.PhotoImage(self.img)
        self.label1.configure(image=photo)
        self.label1.image = photo  # keep a reference!

    def onOpen(self):
        ftypes = [('Image Files', '*.tif *.jpg *.png')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        filename = dlg.show()
        self.fn = filename
        self.setImage()

    def configure_frame(self):
        self.tool_frame.destroy()
        self.tool_frame = tk.Frame(self, border=25, bd=2, relief=tk.RAISED)
        self.tool_frame.grid(row=1, column=0, padx=10, sticky=tk.N)

    def onScale(self):
        self.configure_frame()
        self.slSc = tk.Scale(self.tool_frame, label="scale", from_=5, to=200, orient=tk.HORIZONTAL, command=self.Scale,
                             length=100,
                             width=10, sliderlength=10)
        self.slSc.pack(anchor=tk.N)
        self.button_sc_apl = tk.Button(self.tool_frame, text="Применить", command=self.apply)
        self.button_sc_apl.pack(anchor=tk.N, pady=10)

    def Scale(self, sc):
        self.var_sc = int(sc)
        self.updScale()

    def updScale(self):
        self.I_sc = self.I
        width = int(self.I_sc.shape[1] * self.var_sc / 100)
        height = int(self.I_sc.shape[0] * self.var_sc / 100)
        dim = (width, height)
        resized = cv2.resize(self.I_sc, dim, interpolation=cv2.INTER_AREA)
        self.I_sc = Image.fromarray(np.uint8(resized))
        photo = ImageTk.PhotoImage(self.I_sc)
        self.label2.configure(image=photo)
        self.label2.image = photo

    def apply(self):
        photo = ImageTk.PhotoImage(self.I_sc)
        self.label1.configure(image=photo)
        self.label1.image = photo
        self.I = np.asarray(self.I_sc)

    def onHough(self):
        self.configure_frame()
        self.HoSc1 = tk.Scale(self.tool_frame, label="minDist", from_=1, to=100, orient=tk.HORIZONTAL,
                              command=self.adjDis,
                              length=200, width=10, sliderlength=15)
        self.HoSc1.pack(anchor=tk.W)
        self.HoSc2 = tk.Scale(self.tool_frame, label="Inv.rat.acc.res.", from_=0.2, to=50.0, orient=tk.HORIZONTAL,
                              command=self.adjAcc, length=200, width=10, sliderlength=15, resolution=0.2)
        self.HoSc2.pack(anchor=tk.W)
        self.HoSc3 = tk.Scale(self.tool_frame, label="minRadius", from_=1, to=20, orient=tk.HORIZONTAL,
                              command=self.adjRad1,
                              length=200, width=10, sliderlength=15)
        self.HoSc3.pack(anchor=tk.W)
        self.HoSc4 = tk.Scale(self.tool_frame, label="maxRadius", from_=0, to=100, orient=tk.HORIZONTAL,
                              command=self.adjRad2,
                              length=200, width=10, sliderlength=15)
        self.HoSc4.pack(anchor=tk.W)
        self.HoSc5 = tk.Scale(self.tool_frame, label="higher_thresh", from_=1, to=100, orient=tk.HORIZONTAL,
                              command=self.adjPar1,
                              length=200, width=10, sliderlength=15)
        self.HoSc5.pack(anchor=tk.W)
        self.HoSc6 = tk.Scale(self.tool_frame, label="acc_thresh", from_=1, to=255, orient=tk.HORIZONTAL,
                              command=self.adjPar2,
                              length=200, width=10, sliderlength=15)
        self.HoSc6.pack(anchor=tk.W)

    def adjDis(self, varH):
        self.hv = int(varH)
        print(self.hv)
        self.Houghh()

    def adjAcc(self, ac):
        self.acc = float(ac)
        print(self.acc)
        self.Houghh()

    def adjRad1(self, min_rad):
        self.minRadius = int(min_rad)
        print(self.minRadius)
        self.Houghh()

    def adjRad2(self, max_rad):
        self.maxRadius = int(max_rad)
        print(self.maxRadius)
        self.Houghh()

    def adjPar1(self, par1):
        self.param1 = int(par1)
        print(self.param1)
        self.Houghh()

    def adjPar2(self, par2):
        self.param2 = int(par2)
        print(self.param2)
        self.Houghh()

    def Houghh(self):
        self.I10 = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        # self.I10 = self.I
        self.output = self.I.copy()
        # self.output = cv2.merge([self.I,self.I,self.I])
        self.circles = cv2.HoughCircles(self.I10, cv2.HOUGH_GRADIENT, dp=self.acc, minDist=self.hv, param1=self.param1,
                                        param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)
        if self.circles is not None:
            self.circles = np.round(self.circles[0, :]).astype("int")
            print(len(self.circles))
            for (x, y, r) in self.circles:
                cv2.circle(self.output, (x, y), r, (0, 255, 0), 4)
        self.im = Image.fromarray(np.uint8(self.output))
        photo10 = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo10)
        self.label2.image = photo10

    def on_gauss(self):
        self.I7 = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        self.I7 = cv2.GaussianBlur(self.I7, (9, 9), 0)
        self.im = Image.fromarray(np.uint8(self.I7))
        photo7 = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo7)
        self.label2.image = photo7

    def on_median(self):
        self.I8 = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        self.I8 = cv2.medianBlur(self.I8, 7)
        self.im = Image.fromarray(np.uint8(self.I8))
        photo8 = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo8)
        self.label2.image = photo8

    def on_thresh_bin(self):

        def adj_thresh(val1):
            self.threshB = int(val1)
            self.ThreshBin()

        def adj_max_val(val2):
            self.maxval = int(val2)
            self.ThreshBin()

        self.configure_frame()
        self.BinSc1 = tk.Scale(self.tool_frame, label="thresh", from_=0, to=255, orient=tk.HORIZONTAL,
                               command=adj_thresh, length=200,
                               width=10, sliderlength=15)
        self.BinSc1.pack(anchor=tk.W)
        self.BinSc2 = tk.Scale(self.tool_frame, label="max_val", from_=0, to=255, orient=tk.HORIZONTAL,
                               command=adj_max_val, length=200,
                               width=10, sliderlength=15)
        self.BinSc2.pack(anchor=tk.W)

        def change():
            self.var_thresh_bin = var.get()
            self.ThreshBin()

        var = tk.IntVar()
        var.set(0)
        self.bin = tk.Radiobutton(self.tool_frame, text="THRESH_BINARY",
                                  variable=var, value=0)
        self.bin_inv = tk.Radiobutton(self.tool_frame, text="THRESH_BINARY_INV",
                                      variable=var, value=1)
        self.tosero = tk.Radiobutton(self.tool_frame, text="THRESH_TOZERO",
                                     variable=var, value=3)
        self.otsu = tk.Radiobutton(self.tool_frame, text="THRESH_OTSU",
                                   variable=var, value=8)
        self.button_thresh_bin = tk.Button(self.tool_frame, text="Изменить",
                                           command=change)
        self.bin.pack()
        self.bin_inv.pack()
        self.tosero.pack()
        self.otsu.pack()
        self.button_thresh_bin.pack()

    def ThreshBin(self):
        img = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        self.thresh_img = cv2.threshold(img, self.threshB, self.maxval, type=self.var_thresh_bin)[1]
        im = Image.fromarray(np.uint8(self.thresh_img))
        photo12 = ImageTk.PhotoImage(im)
        self.label2.configure(image=photo12)
        self.label2.image = photo12

    def onCanny(self):
        self.I9 = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        self.I9 = cv2.Canny(self.I9, 20, 100)
        self.im = Image.fromarray(np.uint8(self.I9))
        photo9 = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo9)
        self.label2.image = photo9

    def onClahe(self):
        self.configure_frame()
        self.I6 = self.I
        self.lab = cv2.cvtColor(self.I6, cv2.COLOR_BGR2LAB)
        self.clSc1 = tk.Scale(self.tool_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.adjLimit, length=200,
                              width=10,
                              sliderlength=15)
        self.clSc1.pack(anchor=tk.W)

    def adjLimit(self, varr):
        l, a, b = cv2.split(self.lab)
        self.v = int(varr)
        self.clahe = cv2.createCLAHE(clipLimit=self.v, tileGridSize=(8, 8))
        self.cl = self.clahe.apply(l)
        self.limg = cv2.merge((self.cl, a, b))
        self.final = cv2.cvtColor(self.limg, cv2.COLOR_LAB2BGR)
        self.im = Image.fromarray(np.uint8(self.final))
        photo6 = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo6)
        self.label2.image = photo6

    # --------------------TRESHOLD-------------------------------------

    def on_threshold(self):

        def adj1(h1):
            self.hvalue1 = int(h1)
            self.thresh()

        def adj2(s1):
            self.hvalue2 = int(s1)
            self.thresh()

        def adj3(v1):
            self.hvalue3 = int(v1)
            self.thresh()

        def adj4(h2):
            self.hvalue4 = int(h2)
            self.thresh()

        def adj5(s2):
            self.hvalue5 = int(s2)
            self.thresh()

        def adj6(v2):
            self.hvalue6 = int(v2)
            self.thresh()

        self.configure_frame()
        self.Btn1 = tk.Button(self.tool_frame, text="Show", command=self.thresh)
        self.Btn1.pack(side=tk.TOP)
        self.Btn2 = tk.Button(self.tool_frame, text="Cancel", command=self.onPos)
        self.Btn2.pack(side=tk.TOP)
        self.brgSc1 = tk.Scale(self.tool_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=adj1, length=200,
                               width=10, sliderlength=15)
        self.brgSc1.pack(anchor=tk.S)
        self.brgSc2 = tk.Scale(self.tool_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=adj2, length=200,
                               width=10, sliderlength=15)
        self.brgSc2.pack(anchor=tk.S)
        self.brgSc3 = tk.Scale(self.tool_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=adj3, length=200,
                               width=10, sliderlength=15)
        self.brgSc3.pack(anchor=tk.S)
        self.brgSc4 = tk.Scale(self.tool_frame, from_=255, to=0, orient=tk.HORIZONTAL, command=adj4, length=200,
                               width=10, sliderlength=15)
        self.brgSc4.pack(anchor=tk.S)
        self.brgSc5 = tk.Scale(self.tool_frame, from_=255, to=0, orient=tk.HORIZONTAL, command=adj5, length=200,
                               width=10, sliderlength=15)
        self.brgSc5.pack(anchor=tk.S)
        self.brgSc6 = tk.Scale(self.tool_frame, from_=255, to=0, orient=tk.HORIZONTAL, command=adj6, length=200,
                               width=10, sliderlength=15)
        self.brgSc6.pack(anchor=tk.S)

    def thresh(self):
        hsv = cv2.cvtColor(self.I, cv2.COLOR_BGR2HSV)
        h_min = np.array((self.hvalue1, self.hvalue2, self.hvalue3), np.uint8)
        h_max = np.array((self.hvalue4, self.hvalue5, self.hvalue6), np.uint8)
        print("h_min", h_min)
        print("h_max", h_max)
        thresh = cv2.inRange(hsv, h_min, h_max)
        im = Image.fromarray(np.uint8(thresh))
        photo = ImageTk.PhotoImage(im)
        self.label2.configure(image=photo)
        self.label2.image = photo

    def saveImage(self):
        self.edge = self.im
        filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        if not filename:
            return
        self.edge.save(filename)

    def onNeg(self):
        # Image Negative Menu callback
        self.I2 = 255 - self.I
        self.im = Image.fromarray(np.uint8(self.I2))
        photo_neg = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo_neg)
        self.label2.image = photo_neg

    def onGray(self):
        img = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        self.im = Image.fromarray(np.uint8(img))
        photo_gray = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo_gray)
        self.label2.image = photo_gray

    def onCnt(self):
        self.configure_frame()
        self.cntSc1 = tk.Scale(self.tool_frame, label="len", from_=5, to=10, orient=tk.HORIZONTAL, command=self.adjlen,
                               length=200, width=10,
                               sliderlength=15)
        self.cntSc1.pack(anchor=tk.W)
        self.cntSc2 = tk.Scale(self.tool_frame, label="area min", from_=5, to=1000, orient=tk.HORIZONTAL,
                               command=self.adjArea1, length=200, width=10,
                               sliderlength=15)
        self.cntSc2.pack(anchor=tk.W)
        self.cntSc3 = tk.Scale(self.tool_frame, label="area max", from_=100, to=80000, orient=tk.HORIZONTAL,
                               command=self.adjArea2, length=200, width=10,
                               sliderlength=15)
        self.cntSc3.pack(anchor=tk.W)
        self.cntSc4 = tk.Scale(self.tool_frame, label="diam", from_=10, to=1000, orient=tk.HORIZONTAL,
                               command=self.adjDiam, length=200, width=10,
                               sliderlength=15)
        self.cntSc4.pack(anchor=tk.W)
        # self.cntSc5 = tk.Scale(cntTk, from_=10, to=100, orient=tk.HORIZONTAL, command=self.adjThreshc, length=200, width=10,
        #                         sliderlength=15)
        # self.cntSc5.pack(anchor=tk.W)

    def adjlen(self, lencnt):
        self.lencnt1 = int(lencnt)
        self.on_contours()

    def adjArea1(self, areacnt1):
        self.areacnt1 = int(areacnt1)
        self.on_contours()

    def adjArea2(self, areacnt2):
        self.areacnt2 = int(areacnt2)
        self.on_contours()

    def adjDiam(self, diamcnt):
        self.diamcnt = int(diamcnt)
        self.on_contours()

    # def adjThreshc(self, Threshcnt):
    #     self.Threshcnt = int(Threshcnt)
    #     print(self.Threshcnt)
    #     self.onContours()

    def on_contours(self):
        img = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(img, 0, 255, type=8)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_с = self.I.copy()
        total = 0
        l = []
        circlesC = None
        contour_list = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(cnt)
            if (len(approx) > self.lencnt1) & (self.areacnt1 < area < self.areacnt2):
                contour_list.append(cnt)
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                d = ma / 2
                rr = MA / ma
                l.append(d)
                print(d)
                if (5 < d < self.diamcnt) & (rr > 0.7):
                    cv2.ellipse(img_с, ellipse, (0, 10, 255), 2)
                total += 1
        im2 = Image.fromarray(np.uint8(img_с))
        im = Image.fromarray(np.uint8(thresh))
        photo1 = ImageTk.PhotoImage(im)
        photo2 = ImageTk.PhotoImage(im2)
        self.label1.configure(image=photo1)
        self.label1.image = photo1
        self.label2.configure(image=photo2)
        self.label2.image = photo2
        print('number', total)

    def onPos(self):
        I = self.I
        self.im = Image.fromarray(np.uint8(I))
        photo = ImageTk.PhotoImage(self.im)
        self.label2.configure(image=photo)
        self.label2.image = photo
        self.label1.configure(image=photo)
        self.label1.image = photo


def main():
    root = tk.Tk()
    APP(root)
    root.geometry("320x240")
    root.mainloop()


if __name__ == '__main__':
    main()
