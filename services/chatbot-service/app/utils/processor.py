import re

class Preprocessor:
    def loadText(files):
        '''
        This function is used to load the data (txt format)

        Parameters
        ----------
        files: list
            List of files' name to load the data

        Returns
        -------
        One string contains all the texts from data
        '''
        text = ''
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                text += re.sub(r"\s+", " ", f.read().strip()) + ' '
        return text
    