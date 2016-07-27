import sys

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    From:
    http://stackoverflow.com/a/3041990/279858
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


class TablePrinter(object):
    '''
    There are many python ascii table printers on the web, but this is specific
    for a table which is live updated with new numbers as they come in.
    Prints from a list of dictionaries.

    Example usage:

    results = [{'epoch':1, 'train_loss': 0.3},
               {'epoch':2, 'train_loss': 0.25, 'val_loss': 0.35},
               {'epoch':3, 'train_loss': 0.21},
               {'epoch':4, 'train_loss': 0.20, 'val_loss': 0.32}]

    col_headings = ['epoch', 'train_loss', 'val_loss']
    tableprinter = TablePrinter(col_headings)
    tableprinter.print_header()

    for result in results:
        tableprinter.print_row(result)

    '''
    def __init__(self, col_headings, row_widths=10, sep='|',
            ignore_missing=True):
        self.col_headings = col_headings
        self.n_cols = len(col_headings)
        self.sep = sep
        self.ignore_missing = ignore_missing

        if hasattr(row_widths, '__iter__'):
            self.row_widths = row_widths
        else:
            self.row_widths = [row_widths] * len(col_headings)

        assert(len(self.row_widths) == len(col_headings))

    def print_header(self):
        '''Prints column headings, modifying the strings to look a bit nicer'''

        r = '\n' + self.sep + ' '

        for heading, width in zip(self.col_headings, self.row_widths):
            title = heading.strip().replace('_', ' ').capitalize()
            r += title.ljust(width) +  ' ' + self.sep + ' '

        r += '\n' + '-' * (sum(self.row_widths) + 3 * len(self.row_widths) + 1)
        print r

    def _str_repr(self, item):
        '''
        Returns a string representation of an item, limited to a max length
        (unlike e.g. 'str'). There are some design decisions made here, which
        limit the generalisability of this whole thing. For logging in
        neural network training, though, this should be fine
        '''
        if type(item) is str:
            return item
        elif type(item) is float:
            return "%0.4f" % item
        elif type(item) is int:
            return "%6d" % item
        elif item is None:
            return "None"
        else:
            return str(item)

    def print_row(self, data, print_header=False):
        '''Prints a single row of results to screen'''

        if print_header:
            self.print_header()

        r = self.sep + ' '
        for heading, width in zip(self.col_headings, self.row_widths):
            if self.ignore_missing and heading not in data:
                item = '---'
            else:
                item = data[heading]

            r += self._str_repr(item).center(width) + ' | '

        print r


class Logger(object):
    '''
    Allows printing to both screen and to a file.
    http://stackoverflow.com/a/14906787/279858

    Usage:
    >> import sys
    >> sys.stdout = Logger('log.txt')
    '''
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        pass


def color_grid_vis(X, (nh, nw), save_path=None):
    '''https://github.com/Newmu/dcgan_code'''
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img
