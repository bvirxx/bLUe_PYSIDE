from win32com.client import Dispatch, VARIANT
from pythoncom import VT_VARIANT
import collections

from matplotlib import pyplot as plt


appObj = Dispatch("Photoshop.Application")
#appObj.Documents.Add()

def variant (data) :
    return VARIANT (VT_VARIANT ,data )

def vararr ( *data ) :
    if len(data) == 1 and isinstance ( data, collections.Iterable ) :
        data = data [0]
    return map(variant, data)

docObj = appObj.ActiveDocument
docObj.ColorSamplers.RemoveAll()

sampler=docObj.colorSamplers.Add([0,0])

docObj.Selection.SelectAll()

c=sampler.color

print c.lab.l , c.lab.a, c.lab.b
layer = docObj.ArtLayers[0]
#layer.AdjustBrightnessContrast( 20, -15)

#docObj.ActiveChannels = (docObj.Channels[0] , docObj.Channels[1] , docObj.Channels[2] , docObj.Channels[3])

hist = docObj.histogram

plt.plot(hist)
plt.show()

print hist

layer.AdjustCurves ( vararr ( [ [0,0] , [255,200] ]))

hist = docObj.histogram

print hist


