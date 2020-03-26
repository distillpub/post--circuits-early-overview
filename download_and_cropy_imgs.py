from collections import defaultdict
from lucid.misc.io import load, save
import numpy as np
import os


layer_sizes = {
    "conv2d0" : 64,
    "conv2d1" : 64,
    "conv2d2" : 192,
    "mixed3a" : 256,
    "mixed3b" : 480,
    "mixed4a" : 508,
    "mixed4b" : 512,
    "mixed4c" : 512,
    "mixed4d" : 528,
}

W_dict = {"mixed3a": 60, "mixed3b" : 60, "mixed4a": 100, "mixed4b" : 110, "mixed4c" : 120, "mixed4d" : 130}

def vis_url(layer_name, n):

  if layer_name == "localresponsenorm0" or layer_name == "maxpool0":
    layer_name = "conv2d0"

  if layer_name == "localresponsenorm1":
    layer_name = "conv2d02"

  if layer_name == "conv2d0":
    pass
    return "https://storage.googleapis.com/clarity-public/colah/experiments/aprox_weights_1/%s_%s.png" % (layer_name, n)
  elif layer_name in [ "conv2d1", "conv2d2"]:
    return "https://storage.googleapis.com/clarity-public/colah/experiments/aprox_weights_1/%s_%s.png" % (layer_name, n)
  else:
    return "https://openai-clarity.storage.googleapis.com/model-visualizer%2F1556758232%2FInceptionV1%2Ffeature_visualization%2Falpha%3DFalse%26layer_name%3D"+layer_name+"%26negative%3DFalse%26objective_name%3Dneuron%2Fchannel_index="+str(n)+".png"


for layer in list(layer_sizes.keys())[3:5]:
  W = W_dict[layer]
  for unit in range(layer_sizes[layer]):
    url = vis_url(layer, unit)
    img = load(url)
    D = (img.shape[0] - W)//2
    if layer in ["mixed3a", "mixed3b"]:
      D += 5
    img = img[D:D+W, D:D+W]
    save(img, "public/images/neuron/%s_%s.jpg" % (layer, unit))
    print(".", end="")
    if (unit+1) % 20 == 0: print("")
  print("\n")
