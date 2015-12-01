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
    def __init__(self, col_headings, row_width=10, sep='|',
            ignore_missing=True):
        self.col_headings = col_headings
        self.row_width = row_width
        self.n_cols = len(col_headings)
        self.sep = sep
        self.ignore_missing = ignore_missing

    def print_header(self):
        '''Prints column headings, modifying the strings to look a bit nicer'''

        r = '\n' + self.sep + ' '

        for heading in self.col_headings:
            title = heading.strip().replace('_', ' ').capitalize()
            r += title.ljust(self.row_width) +  ' ' + self.sep + ' '

        r += '\n' + '-' * ((self.row_width + 3) * self.n_cols + 1)
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

    def print_row(self, data, print_header=False):
        '''Prints a single row of results to screen'''

        if print_header:
            self.print_header()

        r = self.sep + ' '
        for heading in self.col_headings:
            if self.ignore_missing and heading not in data:
                item = '---'
            else:
                item = data[heading]

            r += self._str_repr(item).ljust(self.row_width) + ' | '

        print r


def training_graph(graph_savepath, run_results, labels_to_plot):
    '''
    Plot and save a training graph showing losses and/or accuracies
    Losses and accuracies are plotted in separate subplots
    Labels are assumed to be of the form <datasetname>_<T>, where T is either
    ls (loss) or ac (accuracy): e.g. "synth_ls".
    '''
    plt.clf()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    pal = sns.color_palette()
    cols_dict = {
        'train': pal[0], 'synth': pal[1], 'nhm': pal[2], 'ispot': pal[3]}

    for label in labels_to_plot:

        this_x = []
        this_y = []

        for result in run_results:
            # some things aren't in each iteration of the loop, e.g. val_loss
            if label in result:
                this_y.append(result[label])
                this_x.append(result['epoch'])

        nm = label.split('_')[0]

        if label.endswith('ls'):
            ax1.semilogy(this_x, this_y, color=cols_dict[nm], label=label)
        elif label.endswith('ac'):
            ax2.plot(this_x, this_y, color=cols_dict[nm], label=label)
        else:
            raise Exception("Don't know this label...%s" % label)

    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax1.legend(loc='best')
    ax2.legend(loc='best')