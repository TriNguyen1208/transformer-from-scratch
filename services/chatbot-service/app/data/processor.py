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
        One string contains all the texts in files
        '''

        text = ''

        for file in files:
            with open(file, 'r', encoding='utf-8') as f_r:
                # text += re.sub(r'\d+', '', f_r.read()).replace('\n', ' ').replace('\t', ' ') + ' '
                text += f_r.read().replace('\n', ' ').replace('\t', ' ') + ' '

        return text

    def splitIntoSentences(text):
        '''
        This function is used to split the whole text into multiple sentences

        Parameters
        ----------
        text: string
            string contains text top split

        Returns
        -------
        One list contains all the sentences in text
        '''
        
        return re.split(r'(?<=[.!?:])\s+', text)
