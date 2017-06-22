

end_line = 5000000

in_file_name = r'\\10.141.208.22\data\baidubaike\baike_content.txt'
out_file_name = r'C:\Users\cjd\Desktop\baike_corpus.txt'

in_file = open(in_file_name, 'r', encoding='utf-8')
out_file = open(out_file_name, 'a', encoding='utf-8')
print(out_file)

text_line = in_file.readline()
line_index = 1


while text_line:
    if line_index > end_line:
        break
    res = text_line.split('\t')
    if len(res) > 2:
        sentence = res[2].strip()
        for clause in sentence.split(' '):
            if len(clause) > 20 and '是' in clause:
                out_file.write(clause+'\n')

    text_line = in_file.readline()
    line_index += 1
    if line_index % 2000 == 0:
        print(line_index)