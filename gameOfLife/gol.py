import cv2 
import numpy as np
import numba
import time

screenHeight=500
screenWidth=600
cellsX=60
cellsY=50

tscreenHeight=screenHeight-(screenHeight%cellsY)
cellHeight=tscreenHeight//cellsY
tscreenWidth=screenWidth-(screenWidth%cellsX)
cellWidth=tscreenWidth//cellsX
canvas=np.zeros((tscreenHeight,tscreenWidth),np.uint8)
cells=np.zeros((cellsY,cellsX),dtype=bool)
print(canvas.shape)
def drawCell(cx,cy,val):
    startX=cx*cellWidth
    startY=cy*cellHeight
    canvas[startY:startY+cellHeight,startX:startX+cellWidth]=val
    #print('afjdiajsfoj',canvas.any(),canvas[startY:startY+cellHeight,startX:startX+cellWidth])

def countNeighbors1(cells, x,y):
    deltas=[-1,1]
    rows, cols=cells.shape
    count=0
    count+=(x+1<cols and cells[y,x+1])
    count+=(x-1>=0 and cells[y,x-1])
    count+=(y+1<rows and cells[y+1,x])
    count+=(y-1>=0 and cells[y-1,x])
    return count

def countNeighbors2(cells, x,y):
    deltas=[-1,0,1]
    count=0
    for dx in deltas:
        for dy in deltas:
            newX=x+dx
            newY=y+dy
            if newX>=0 and newX<cellsX and newY>=0 and newY<cellsY and (not (dx==0 and dy==0)) and cells[newY,newX]:
                count+=1
    return count
            

def update(cells):
    newCells=np.zeros_like(cells)
    rows, cols=cells.shape
    for i in range(rows):
        for j in range(cols):
            isLive=cells[i,j]
            neighbors=countNeighbors2(cells,j,i)
            if (isLive and (neighbors==3 or neighbors==2)) or (not isLive and neighbors==3):
                #print('update true')
                newCells[i,j]=True
            else:
                newCells[i,j]=False
    return newCells

def shift(cells, dx=None, dy=None):
    newcells=np.zeros_like(cells)
    if dx is not None:
        #shift to the left or right
        #left is positive and means a new column is created on the left
        if dx>0:
            shifted=cells[:,:cellsX-dx]
            newcells[:,dx:]=shifted
        elif dx<0:
            shifted=cells[:,-dx:]
            newcells[:,:cellsX+dx]=shifted
    if dy is not None:
        #shift up or down
        #up is positive where a new row is created at the top
        if dy>0:
            shifted=cells[:cellsY-dy,:]
            newcells[dy:,:]=shifted
        elif dy<0:
            shifted=cells[-dy:,:]
            newcells[:cellsY+dy,:]=shifted
    return newcells

def smartDraw(oldCells, newCells):
    for row in range(cellsY):
        for col in range(cellsX):
            if oldCells[row,col]!=newCells[row,col]:
                #print('drawing on canvas')
                drawCell(col, row, newCells[row,col]*255)

def getMouseCellPos(event,x,y,flags,param):
    cellPosX=x//cellWidth
    cellPosY=y//cellHeight
    if event==1:
        #print(cellPosX, cellPosY)
        cells[cellPosY,cellPosX]=True
        drawCell(cellPosX, cellPosY, 255)
    else:
        pass
        #canvas=cv2.rectangle(canvas, (cellPosX, cellPosY),(cellPosX+cellWidth, cellPosY+cellHeight),255,3)

isPaused=True
shiftAmt=1
cv2.namedWindow("life")
cv2.setMouseCallback('life',getMouseCellPos)
backup=cells.copy()
while 1:
    cv2.imshow("life",canvas)
    k=cv2.waitKey(1)
    if k==ord('p'):
        isPaused= not isPaused
        print("paused: ",isPaused)
    if k==ord('q'):
        break
    if not isPaused:
        backup=cells.copy()
        cells=update(cells)
        if k==ord('a'):#move left
            cells=shift(cells,dx=shiftAmt)
        if k==ord('d'):
            cells=shift(cells,dx=-shiftAmt)
        if k==ord('w'):
            cells=shift(cells, dy=shiftAmt)
        if k==ord('s'):
            cells=shift(cells, dy=-shiftAmt)
        smartDraw(backup,cells)
    time.sleep(.05)
cv2.destroyAllWindows()
