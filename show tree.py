#3/6/19 ML wednesday

#given by sir
pip install graphviz
export_graphviz(tree, out_file="tree.dot", class_names = ["maligent", "benign"],
feature_names = cancer.feature_names, impurity = False, filled = True)


import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


