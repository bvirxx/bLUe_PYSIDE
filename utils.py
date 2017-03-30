"""
This File is part of bLUe software.

Copyright (C) 2017  Bernard Virot <bernard.virot@libertysurf.fr>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Lesser Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

from PySide.QtGui import QListWidget, QListWidgetItem
from PySide.QtCore import Qt

class optionsWidget(QListWidget) :
    """
    Displays a list of options with checkboxes.
    The choices can be mutually exclusive (default) or not
    exclusive.
    """

    def __init__(self, options=[], exclusive=True):
        """
        :param options: list of strings
        :param exclusive: boolean
        """
        super(optionsWidget, self).__init__()
        self.items = {}
        for option in options:
            listItem = QListWidgetItem(option, self)
            listItem.setCheckState(Qt.Unchecked)
            listItem.mySelectedAttr = False
            self.addItem(listItem)
            self.items[option] = listItem
        self.setMinimumWidth(self.sizeHintForColumn(0))
        self.exclusive = exclusive
        self.itemClicked.connect(self.select)
        # selection hook.
        self.onSelect = lambda x : 0

    def select(self, item):
        """
        Mouse click event handler
        :param item:
        """
        for r in range(self.count()):
            currentItem = self.item(r)
            if currentItem is item:
                if self.exclusive:
                    currentItem.setCheckState(Qt.Checked)
                    currentItem.mySelectedAttr = True
                else:
                    currentItem.setCheckState(Qt.Unchecked if currentItem.mySelectedAttr else Qt.Checked)
                    currentItem.mySelectedAttr = not currentItem.mySelectedAttr
            elif self.exclusive:
                currentItem.setCheckState(Qt.Unchecked)
                currentItem.mySelectedAttr = False
            self.onSelect(item)