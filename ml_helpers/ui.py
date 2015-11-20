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
    Prints from a dictionary
    '''
    def __init__(self, col_headings, row_width=10, sep='|'):
        self.col_headings = col_headings
        self.row_width = row_width
        self.n_cols = len(col_headings)
        self.sep = sep

    def print_header(self):
        '''Prints column headings, modifying the strings to look a bit nicer'''

        r = '\n' + self.sep + ' '

        for heading in self.col_headings:
            title = heading.strip().replace('_', ' ').capitalize()
            r += title.ljust(self.row_width) +  ' ' + self.sep + ' '

        r += '\n' + '-' * ((self.row_width + 3) * self.n_cols + 1)
        print r

    def print_row(self, data, print_header=False):
        '''Prints a single row of results to screen'''

        if print_header:
            self.print_header()

        r = self.sep + ' '
        for heading in self.col_headings:
            r += str(data[heading]).ljust(self.row_width) + ' | '

        print r
