#!/Users/majian/anaconda/bin/python

import cv2
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib as mpl

'''use instruction
first: modify labels
function:
    analyze the distrution os image attribute:
        width, height, labels
'''
class DataAnalyze:
    def __init__(self):
        self.labels = {'qwe':0,'asd':1, 'zxc':2, }
        self.nameList = []
        self.attribute = []

    def plotColor(self):
        color = {}
        COLOR = ['red','blue','green','black', 'yello']

        for key in self.labels.keys():
            color[self.labels[key]] = COLOR[self.labels[key]]
        return color

    def getPicList(self,dirName):
        for root,dirNames,fileNames in os.walk(dirName):
            for name in fileNames:
                self.nameList.append(os.path.join(root,name))

    def drawPic(self, attributes,color):
        attributes = np.array(attributes)
        lenLabel = np.max(attributes[:,-1])
        for i in np.arange(lenLabel+1):
            index = np.where(attributes[:,-1] == i)
            plt.scatter(attributes[index,0], attributes[index,1],color = color[i], label = i)
        plt.title('Image Attribute Analyize')
        plt.xlabel('width')
        plt.ylabel('height')
        plt.legend()
        plt.show()

    def getAttributations(self, nameList):
        attributes = []
        for name in nameList:
            if name.split('.')[-1] not in ['jpg','png','JPG']:
                continue
            img = cv2.imread(name)
            if os.path.dirname(name).split('/')[-1] in self.labels.keys():
                attribute = img.shape + (self.labels[os.path.dirname(name).split('/')[-1]],)
                attributes.append(attribute)
        return attributes

    def draw_scatter(self, Path):
        color = self.plotColor()
        self.getPicList(Path)
        attributes = self.getAttributations(self.nameList)
        self.drawPic(attributes,color)

def plotpie():
    def draw_pie(labels,quants):
        # make a square figure
        plt.figure(1, figsize=(6,6))
        # For China, make the piece explode a bit
        expl = [0,0.1,0,0,0,0,0,0,0,0]   
        # Colors used. Recycle if not enough.
        colors  = ["blue","red","coral","green","yellow","orange"]  
        # Pie Plot
        # autopct: format of "percent" string;
        plt.pie(quants, explode=expl, colors=colors, labels=labels, autopct='%1.1f%%',pctdistance=0.8, shadow=True)
        plt.title('Top 10 GDP Countries', bbox={'facecolor':'0.8', 'pad':5})
        plt.show()
        plt.savefig("pie.jpg")
        plt.close()

    # quants: GDP

    # labels: country name

    labels   = ['USA', 'China', 'India', 'Japan', 'Germany', 'Russia', 'Brazil', 'UK', 'France', 'Italy']

    quants   = [15094025.0, 11299967.0, 4457784.0, 4440376.0, 3099080.0, 2383402.0, 2293954.0, 2260803.0, 2217900.0, 1846950.0]

    draw_pie(labels,quants)

if __name__=='__main__':
    #data = DataAnalyze();
    #data.analyzeAndDraw('./')
    plotpie()
