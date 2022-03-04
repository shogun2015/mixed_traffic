from xml.etree.ElementTree import ElementTree,Element

def read_xml(path):
    tree = ElementTree()
    tree.parse(path)
    return tree


def write_xml(tree, path):
    tree.write(path, encoding="utf-8", xml_declaration=True)


def alterDemand(path, probability, icv_ratio):
    tree = read_xml(path)
    root = tree.getroot()
    probability_icv = probability * icv_ratio
    probability_hdv = probability - probability_icv
    if icv_ratio == 1:
        probability_hdv = 0.00000000001
    for flow_node in root.findall('flow'):
        if "HDV" in flow_node.get("id"):
            flow_node.set("probability", str(probability_hdv))
        elif "ICV" in flow_node.get("id"):
            flow_node.set("probability", str(probability_icv))
    write_xml(tree, path)
    print("Alter demand factor to hdv: %s ; icv:%s" % (probability_hdv, probability_icv))