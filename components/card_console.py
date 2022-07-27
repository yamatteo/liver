import contextlib

from ipywidgets import VBox, Output, Layout


class Card(Output):
    def __init__(self):
        super(Card, self).__init__(layout=Layout(border="dotted grey 2px"))

    def close(self):
        self.layout = Layout(border="", padding="2px")


class Console(VBox):
    def insert(self):
        card = Card()
        self.children = (card, *self.children)
        return card

    @contextlib.contextmanager
    def new_card(self):
        card = self.insert()
        with card:
            yield
        card.close()