from collections import defaultdict
import numpy as np


layer_sizes = {
    "conv2d0" : 64,
    "conv2d1" : 64,
    "conv2d2" : 192,
    "mixed3a" : 256,
    "mixed3b" : 480,
    "mixed4a" : 508,
}

def vis_html(layer_name, n, W=120):

  if layer_name == "localresponsenorm0" or layer_name == "maxpool0":
    layer_name = "conv2d0"

  if layer_name == "localresponsenorm1":
    layer_name = "conv2d02"

  if layer_name == "conv2d0":
    pass
    # weight = param['conv2d0_w'][..., n]
    # weight = 0.6*weight / np.abs(param['conv2d0_w']).max() + 0.4*weight / np.abs(weight).max()
    # img_url = _image_url(0.5+0.5*weight, domain=[0,1])
    img_url = "https://storage.googleapis.com/clarity-public/colah/experiments/aprox_weights_1/%s_%s.png" % (layer_name, n)
    img = "<img style='width: 100%%; image-rendering: pixelated;' src='%s'>" % (img_url)
  elif layer_name in [ "conv2d1", "conv2d2"]:
    img_url = "https://storage.googleapis.com/clarity-public/colah/experiments/aprox_weights_1/%s_%s.png" % (layer_name, n)
    img = "<img style='width: 100%%;' src='%s'>" % (img_url)
  else:
    img_url = "https://openai-clarity.storage.googleapis.com/model-visualizer%2F1556758232%2FInceptionV1%2Ffeature_visualization%2Falpha%3DFalse%26layer_name%3D"+layer_name+"%26negative%3DFalse%26objective_name%3Dneuron%2Fchannel_index="+str(n)+".png"
    img = "<img style='margin-left: -%spx; margin-top: -%spx;' src='%s'>" % ((224 - W)//2+0.1*W, (224 - W)//2+0.1*W, img_url)
  img = "<div style='width: %spx; height: %spx; margin-right: 1px; overflow: hidden; display: inline-block;'>%s</div>" % (W, W, img)

  a_url =  "https://storage.googleapis.com/clarity-public/colah/experiments/aprox_weights_1/%s_%s.html" % (layer_name, n)
  #a_url =  "https://storage.googleapis.com/clarity-public/colah/experiments/many-low-level-tuning-curves/%s_%s.html" % (layer_name, n)
  img = "<a href='%s'>%s</a>" % (a_url, img)

  return img



def units_to_width(group, max_rec_height=None):

  max_rec_height = max_rec_height or 10

  group_units = group["units"]

  n_units = len(group_units)
  if n_units >= 5*10 and n_units % 5 == 0 and max_rec_height >= 5:
    target_n_height = 5
  elif n_units > 4*4 and n_units % 4 == 0 and  max_rec_height >= 4:
    target_n_height = 4
  elif n_units <= 3*12 and n_units % 3 == 0 and n_units >= 3*3 and  max_rec_height >= 3:
    target_n_height = 3
  elif n_units <= 2*12 and n_units % 2 == 0 and n_units >= 2*4 and  max_rec_height >= 2:
    target_n_height = 2
  elif n_units >= 4*4 and  max_rec_height >= 4:
    target_n_height = 4
  elif n_units >= 3*3 and  max_rec_height >= 3:
    target_n_height = 3
  elif n_units >= 2*2 and  max_rec_height >= 2:
    target_n_height = 2
  else:
    target_n_height = 1

  n_width = min(len(group_units), (len(group_units)-1)//target_n_height+1)

  comment = group["comment"]

  if comment and n_units == 4:
    n_width = 4
  elif comment and n_width <= 3:
    n_width = 3

  comment_len = len(comment.split(" ")) + (10 if "<br><br>" in comment else 0 )

  if comment_len > 25:
    n_width = max(n_width, 4)
  if comment_len > 40:
    n_width = max(n_width, 5)

  return n_width


def render(groups, max_rec_height=None, show_html=False, priority_filter=lambda x: True):

  group_layer = groups[0]["layer"]


  unused_units = [n
                  for n in range(layer_sizes[group_layer])
                  if not any([n in group["units"] for group in groups])]

  html =""


  def unit_group_ord(group):
    n_units = len(group["units"])
    width = units_to_width(group, max_rec_height=max_rec_height)
    comment = group["comment"]
    return (-group["priority"], -np.ceil(n_units/width), -(n_units + len(comment.split(" "))/4.))


  flex_content  = ""

  for group in sorted(groups, key=unit_group_ord):

    if not priority_filter(group["priority"]): continue

    group_name, group_layer, group_units = group["name"], group["layer"], group["units"]

    W_dict = {"mixed4a": 100}
    W = W_dict[group_layer] if group_layer in W_dict else 60

    n_width = units_to_width(group)

    flex_content += "<div style='flex-basis: %spx; flex-grow: %s; max-width: %spx;' class='group'> <div><h3>%s</h3> <div>%s</div></div><div class='figcaption'>%s</div></div>" % (
        2+max(100, (W+3)*n_width+1),
        len(group_units) + (100 if np.ceil(len(group_units)/n_width) >=5 else 30 if np.ceil(len(group_units)/n_width) >=3 else 0),
        2+max(100, (W+3)*max(n_width, len(group_units))+1),
        "<b>" + group_name.replace(group_layer+"_", "") + "</b> %s%%" % str((1000*len(group_units)//layer_sizes[group_layer])*0.1)[:4],
        "".join(["<div class='neuron'>%s<div class='label'>%s</div></div>" % (vis_html(group_layer, unit, W=W), unit)
          for unit in group_units
      ]),
      group["comment"]
    )

  html += "<figure class='l-screen-inset'><div style='display:flex; flex-wrap: wrap;'>%s</div></figure>" % flex_content

  if unused_units:
    print("unused units:", unused_units)
    html += "<figure class='l-screen-inset'><p>Other Units in " + layer + " (%s%%)</p><br>"  % str((1000*len(unused_units)//layer_sizes[group_layer])*0.1)[:4]
    html += "<div class='group'><div>%s</div></div>" % (
      "".join(["<div class='neuron'>%s<div class='label'>%s</div></div>" % (vis_html(group_layer, unit, W=W), unit)
        for unit in unused_units
      ]))

  html += "</figure><br><br><br><br><br>"

  return html

figure_html = {}

def render_layer(layer, priority_filter=lambda x: True, suffix=""):
  data = eval(open("layer_data/%s.json" % layer, "r").read())
  for x in data:
    if "layer" not in x:
      x["layer"] = layer
    if "priority" not in x:
      x["priority"] = 0
  html = render(data,priority_filter=priority_filter)
  figure_html[layer+suffix] = html
  html = """
  <link rel="stylesheet" type="text/css" href="index.css">
  %s
  """ % html
  if suffix == "":
    out = open("public/%s.html" % layer, "w").write(html)

for layer in layer_sizes.keys():
  print(layer)
  render_layer(layer)

render_layer("mixed3b", priority_filter=lambda p: p >= 1, suffix="_hipri")
render_layer("mixed3b", priority_filter=lambda p: p < 1, suffix="_lowpri")

print([(k, type(figure_html[k])) for k in figure_html])

index_template = open("index_template.html", "r").read()
index_html = index_template.format(**figure_html)
open("public/index.html", "w").write(index_html)
