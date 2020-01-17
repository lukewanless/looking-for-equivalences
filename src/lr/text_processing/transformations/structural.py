def invert(p, h, l):
    """
    transformation that returns h, p, l


    :param p: premise
    :type p: str
    :param h: hypothesis
    :type h: str
    :param l: label
    :type l: str
    :return: new observation (p_new,h_new,l_new)
    :rtype: (str,str,str)
    """
    return h, p, l


def label_internalization(p, h, l):
    """
    transformation that adds the label to the pair (p,h)


    :param p: premise
    :type p: str
    :param h: hypothesis
    :type h: str
    :param l: label
    :type l: str
    :return: new observation (p_new,h_new,l_new)
    :rtype: (str,str,str)
    """
    new_p = p
    new_h = h + " , {} ,".format(l)
    return p, new_h, l