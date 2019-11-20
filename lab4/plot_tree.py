import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    cur, l, r = myTree
    if l is None and r is None:
        return 1
    else:
        return getTreeDepth(l)+getTreeDepth(r)


def getTreeDepth(myTree):
    cur, l, r = myTree
    if l is None and r is None:
        return 1
    else:
        return 1+max(getTreeDepth(l), getTreeDepth(r))


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center",
                        ha="center", rotation=30)


def plotTree(tree, parentPt, nodeTxt=''):
    numLeafs = getNumLeafs(tree)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0/plotTree.totalW, plotTree.yOff)
    cur, left, right = tree
    plotMidText(cntrPt, parentPt, nodeTxt)
    if left is None:
        plotNode(cur, cntrPt, parentPt, decisionNode)
    else:
        plotNode('index: {}'.format(cur[0]) , cntrPt, parentPt, decisionNode)
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    if left is None:
        plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
    if left is not None:
        plotTree(left, cntrPt, '<={}'.format(cur[1]))
        plotTree(right, cntrPt, '>{}'.format(cur[1]))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)  # no ticks
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0))
    plt.show()
