from anytree import Node, RenderTree, find

class TEnode(Node):
    separator = "|"


def create_tetree():
    TE = TEnode("TE")

    ClassI = TEnode("ClassI", parent=TE)
    ClassII = TEnode("ClassII", parent=TE)

    LTR = TEnode("LTR", parent=ClassI)
    Non_LTR = TEnode("Non-LTR", parent=ClassI)

    BelPao = TEnode("Bel-Pao", parent=LTR)
    Copia = TEnode("Copia", parent=LTR)
    Gypsy = TEnode("Gypsy", parent=LTR)
    ERV = TEnode("ERV", parent=LTR)

    ERV1 = TEnode("ERV1", parent=ERV)
    ERV2 = TEnode("ERV2", parent=ERV)
    ERV3 = TEnode("ERV3", parent=ERV)
    ERV4 = TEnode("ERV4", parent=ERV)

    DIRS = TEnode("DIRS", parent=Non_LTR)
    PLE = TEnode("PLE", parent=Non_LTR)
    LINE = TEnode("LINE", parent=Non_LTR)
    SINE = TEnode("SINE", parent=Non_LTR)

    CR1 = TEnode("CR1", parent=LINE)
    I = TEnode("I", parent=LINE)
    Jockey = TEnode("Jockey", parent=LINE)
    L1 = TEnode("L1", parent=LINE)
    R2 = TEnode("R2", parent=LINE)
    RTE = TEnode("RTE", parent=LINE)
    Rex1 = TEnode("Rex1", parent=LINE)


    ID = TEnode("ID", parent=SINE)
    SINE1 = TEnode("SINE1/7SL", parent=SINE)
    SINE2 = TEnode("SINE2/tRNA", parent=SINE)
    SINE3 = TEnode("SINE3/5S", parent=SINE)

    Sub1 = TEnode("Sub1", parent=ClassII)
    Sub2 = TEnode("Sub2", parent=ClassII)

    TIR = TEnode("TIR", parent=Sub1)

    CACTA = TEnode("CACTA", parent=TIR)
    MULE = TEnode("MULE", parent=TIR)
    PIF = TEnode("PIF", parent=TIR)
    TcMar = TEnode("TcMar", parent=TIR)
    hAT = TEnode("hAT", parent=TIR)

    Helitron = TEnode("Helitron", parent=Sub2)
    return TE


def get_all_nodes():
    nodes = []
    for pre, fill, node in RenderTree(create_tetree()):
        # print("%s%s" % (pre, node.name))
        nodes.append(node.name)
    return nodes


def get_leaf_nodes():
    te_root = create_tetree()
    leaf_nodes = te_root.leaves
    nodes = [str(node).split("'")[1][1:].split("|")[-1] for node in leaf_nodes]
    return nodes


def get_parent_nodes():
    nodes = list(set(get_all_nodes()) - set(get_leaf_nodes()))
    return nodes


def find_node_path(node_name):
    TE = create_tetree()
    # find(TE, lambda node: node.name == node_name)
    node_path = find(TE, lambda node: node.name == node_name)
    return str(node_path).split("'")[1][1:].replace("|", "@")

