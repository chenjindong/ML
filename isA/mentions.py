# in_file_name = r'C:\Users\cjd\Desktop\mentions.txt'
in_file_name = r'\\10.141.208.24\data\KBQAdata\mentions.txt'

in_file = open(in_file_name, 'r', encoding='utf-8')
text = in_file.read()
mentions = text.split('\n')
print(len(mentions))

