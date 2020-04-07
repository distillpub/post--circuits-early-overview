from collections import defaultdict
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

def vis_html(layer_name, n, W=None):

  W_dict = {"mixed3a": 60, "mixed3b" : 60, "mixed4a": 100, "mixed4b" : 110, "mixed4c" : 120, "mixed4d" : 130}
  if W is None:
    if W in W_dict:
      W = W_dict
    else:
      W=60

  if layer_name == "localresponsenorm0" or layer_name == "maxpool0":
    layer_name = "conv2d0"

  if layer_name == "localresponsenorm1":
    layer_name = "conv2d02"

  if layer_name in [ "conv2d0", "conv2d1", "conv2d2"]:
    img_url = "images/neuron/%s_%s.png" % (layer_name, n)
    img = "<img style='width: 100%%;' src='%s'>" % (img_url)
  elif layer_name in [ "conv2d0", "conv2d1", "conv2d2", "mixed3a", "mixed3b", "mixed4a"]:
    img_url = "images/neuron/%s_%s.jpg" % (layer_name, n)
    img = "<img style='width: 100%%;' src='%s'>" % (img_url)
  else:
    img_url = "https://openai-clarity.storage.googleapis.com/model-visualizer%2F1556758232%2FInceptionV1%2Ffeature_visualization%2Falpha%3DFalse%26layer_name%3D"+layer_name+"%26negative%3DFalse%26objective_name%3Dneuron%2Fchannel_index="+str(n)+".png"
    img = "<img style='margin-left: -%spx; margin-top: -%spx;' src='%s'>" % ((224 - W)//2+0.1*W, (224 - W)//2+0.1*W, img_url)
  img = "<div style='width: %spx; height: %spx; margin-right: 1px; overflow: hidden; display: inline-block;'>%s</div>" % (W, W, img)

  a_url = "https://storage.googleapis.com/inceptionv1-weight-explorer/%s_%s.html" % (layer_name, n)
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


  # def unit_group_ord(group):
  #   n_units = len(group["units"])
  #   width = units_to_width(group, max_rec_height=max_rec_height)
  #   comment = group["comment"]
  #   return (-group["priority"], -np.ceil(n_units/width), -(n_units + len(comment.split(" "))/4.))

  def unit_group_ord(group):
    n_units = len(group["units"])
    width = units_to_width(group, max_rec_height=max_rec_height)
    comment = group["comment"]
    comment_text = " ".join([str.split(">")[1] if  ">" in str else str for str in comment.split("<")])
    #return (-len(comment_text))
    return (-n_units)

  flex_content  = ""


  for group in sorted(groups, key=unit_group_ord):

    if not priority_filter(group["priority"]): continue

    group_name, group_layer, group_units = group["name"], group["layer"], group["units"]

    W_dict = {"mixed3a": 60, "mixed3b" : 60, "mixed4a": 100, "mixed4b" : 110, "mixed4c" : 120, "mixed4d" : 130}
    W = W_dict[group_layer] if group_layer in W_dict else 60

    n_width = units_to_width(group)

    n_group_units = len(group_units)
    percent = 100*len(group_units)/layer_sizes[group_layer]
    if percent >= 0.6:
      size_str = str(int(percent + 0.5))
    else:
      size_str = str((1000*len(group_units)//layer_sizes[group_layer])*0.1)[:3]

    div_id = "group_"+group_layer+"_"+group_name.lower()
    div_id = div_id.replace(" ","_").replace("-", "_")
    div_id = div_id.replace("_/_","_").replace("/","_").replace(",","")

    group_header = "<a href='#"+div_id+"'><b>" + group_name.replace(group_layer+"_", "") + "</b> %s%%</a>" % size_str
    comment = group["comment"]
    neuron_html = "".join([
      "<div class='neuron'>%s<div class='label'>%s</div></div>"
        % (vis_html(group_layer, unit, W=W), unit)
      for unit in group_units[:]]
      )
    max_height = 2*(W+2)

    if n_group_units > 10:
      collapse_toggles = """
      <div class='figcaption collapse-toggle collapse-reveal' onclick='toggle_collapse()'>Show all {n_group_units} neurons.</div>
      <div class='figcaption collapse-toggle collapse-hide' onclick='toggle_collapse()'>Collapse neurons.</div>""".format(**locals())
    else:
      collapse_toggles = "<div class='figcaption collapse-reveal' style='wdith: 40px; height: 16px;'> </div>"


    flex_content += """
      <div class='group collapsed' id='{div_id}'>
        <div>
          <h3>{group_header}</h3>
          <div class='neuron-container' style='--collaposed-height:{max_height}px'>
            {neuron_html}
          </div>
        </div>
        {collapse_toggles}
        <div class='figcaption'>{comment}</div>
      </div>""".format(**locals())


  #html += "<figure class='l-screen-inset'><div style='display:flex; flex-wrap: wrap; margin-top: 40px; margin-bottom: 40px;'>%s</div></figure>" % flex_content
  html += """
  <figure class='l-screen-inset' style='padding-left: 0px;'>
    <div class='group-container' id='{group_layer}-group-container' >{flex_content}</div>
  </figure>""".format(**locals())


  if unused_units:
    print("unused units:", unused_units[:5], "...")
    html += "<figure class='l-screen-inset'><p>Other Units in " + layer + " (%s%%)</p><br>"  % str((1000*len(unused_units)//layer_sizes[group_layer])*0.1)[:4]
    html += "<div class='group'><div>%s</div></div>" % (
      "".join(["<div class='neuron'>%s<div class='label'>%s</div></div>" % (vis_html(group_layer, unit, W=W), unit)
        for unit in unused_units
      ]))

  html += "</figure><br>"

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
  <title>%s Neuron Groups</title>
  %s
  """ % (layer.replace("mixed","").replace("conv", ""), html)
  if suffix == "":
    out = open("public/%s.html" % layer, "w").write(html)

for layer in layer_sizes.keys():
  print(layer)
  render_layer(layer)

for f in os.listdir("public/images/"):
  if ".svg" not in f: continue
  name = f.split(".")[0]
  key = "images/" + name
  print(key)
  lines = open("public/images/" + f).read().split("\n")
  if "<svg" in lines[0]:
    lines[0] = lines[0].replace(">", "id=\"diagram-%s\">"%name)
  lines[2] = lines[2].replace("clip-path", "--disabled-clip-path")
  text = []
  for line in lines:
    line = line.replace("\"pattern", "\"pattern"+name)
    line = line.replace("#pattern", "#pattern"+name)
    line = line.replace("\"image", "\"image"+name)
    line = line.replace("#image", "#image"+name)
    if ("<rect" in line or "<path" in line) and "id=" in line:
      neuron_id = line.split("id=\"")[1].split("\"")[0]
      if "_" in neuron_id and neuron_id.split("_")[0] in layer_sizes:
        if neuron_id.count("_") > 1:
          neuron_id = neuron_id[:-2]
        url = "neurons/%s.html" % neuron_id
        #pattern_n = line.split("#pattern")[1].split(")")[0]
        text.append("<a href='%s'>" % url)
        text.append(line)
        text.append("</a>")
      else:
        text.append(line)
    else:
      text.append(line)
  figure_html[key] = "\n".join(text)

for layer in layer_sizes:
  for unit in range(layer_sizes[layer]):
    name = layer.replace("mixed","").replace("conv2d","")
    a_url =  "https://storage.googleapis.com/clarity-public/colah/experiments/aprox_weights_1/%s_%s.html" % (layer, unit)
    if layer not in ["mixed3a", "mixed3b", "mixed4a"]:
      figure_html["neuron/%s/%s" %(name, unit)] = "<a href=\"%s\"><span>%s:%s</span></a>" % (a_url, name, unit)
    else:
      img_url = "images/neuron/%s_%s.jpg" % (layer, unit)
      figure_html["neuron/%s/%s" %(name, unit)] = "<a href=\"%s\" style=\"border-bottom: none; \"><span style=\"display:inline-block; background: #F5F5F9; border-radius: 3px; padding-left: 2px; height:20px;\"><span>%s:%s</span> <img src=\"%s\" style=\"width:20px; border-radius: 0px 3px 3px 0px; margin-bottom: -4px; margin-left: -2px; display: inline-block;\"></img></span></a>" % (a_url, name, unit, img_url)

render_layer("mixed3b", priority_filter=lambda p: p >= 1, suffix="_hipri")
render_layer("mixed3b", priority_filter=lambda p: p < 1, suffix="_lowpri")

#print([(k, type(figure_html[k])) for k in figure_html])

index_template = open("index_template.html", "r").read()
index_html = index_template.format(**figure_html)
open("public/index.html", "w").write(index_html)
